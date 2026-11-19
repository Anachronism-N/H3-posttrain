"""One-shot worker03 recovery of unclaimed original indices; no transfer/retry."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path('/tmp/ms_pose2d_draft_20261026T1427Z/launcher')
CANDIDATE = ROOT / 'recovery03_unclaimed_v1'
WORKER = 'a100-extra-03'
SOURCES = {
    'unified_lane_rollover_v1.py': '95eff3b6af6104d090b318647bb6442aa95f6b536c6beb4349ff33f8138af1b9',
    'unified_access_transport_rollover_v1.py': 'c8d36e48e930b5dbd235bb21541fe05adc4b6307060bf2bb67658f5f7ff35406',
    'unified_access_transport_v3.py': 'c946aec60e1498bbd48755d1ae783bbfc0955ec50b444434def841498c2a9f0d',
}
ORIGINAL_IDENTITIES = ((136955, 516047987), (137076, 516047992),
                       (137181, 516047997), (137238, 516048001))
SAFE = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]*')
CLAIM_KEYS = {'edition', 'worker', 'gpu', 'assignment_sha256', 'assignment_index',
              'source_row', 'split', 'id', 'pid', 'owner_pid', 'start_ticks',
              'claimed_unix', 'automatic_retry', 'run_id'}


def stable_read(path, limit=64 * 2**20):
    path = Path(path)
    if path.resolve(strict=True) != path:
        raise PermissionError('redirected metadata path')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise PermissionError('metadata size/type')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(limit + 1)
        after, named = os.fstat(fd), path.stat()
        fields = ('st_ino', 'st_dev', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        if len(raw) != before.st_size or any(getattr(before, k) != getattr(s, k)
                                          for s in (after, named) for k in fields):
            raise PermissionError('metadata inode/bytes changed while reading')
        seal = {'sha256': hashlib.sha256(raw).hexdigest(), 'inode': before.st_ino,
                'device': before.st_dev, 'bytes': before.st_size}
        return raw, seal
    finally:
        os.close(fd)


def require_seal(path, expected):
    raw, actual = stable_read(path)
    if actual != expected:
        raise PermissionError('preserved metadata inode/hash changed: ' + str(path))
    return raw


def load_module(name, path, sha):
    raw, seal = stable_read(path)
    if seal['sha256'] != sha:
        raise PermissionError('frozen executable SHA changed')
    module = importlib.util.module_from_spec(importlib.util.spec_from_file_location(name, path))
    # Compile the verified bytes directly; never accept a timestamp-valid stale pyc.
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    require_seal(path, seal)
    sys.modules[name] = module
    return module


base = load_module('recovery03_frozen_lane', ROOT / 'unified_lane_rollover_v1.py',
                   SOURCES['unified_lane_rollover_v1.py'])
transport = load_module('recovery03_frozen_transport', ROOT / 'unified_access_transport_rollover_v1.py',
                        SOURCES['unified_access_transport_rollover_v1.py'])
old_transport = load_module('unified_access_transport_v3', ROOT / 'unified_access_transport_v3.py',
                            SOURCES['unified_access_transport_v3.py'])


def fixed_args(gpu):
    return SimpleNamespace(root=ROOT, candidate=CANDIDATE, worker=WORKER, gpu=gpu,
        ticket=str(base.COMMON / 'assignments' / (WORKER + '.ticket.json')),
        manifest=str(base.COMMON / 'assignments' / (WORKER + '.jsonl')),
        queue=str(base.COMMON / (WORKER + '.queue.json')),
        previous_progress=ROOT / 'logs' / f'unified_gpu{gpu}_progress.json',
        code_root=ROOT / 'unified_code_snapshot_v1', scail_root=ROOT / 'unified_scail_local_v1',
        beat_checkpoint=ROOT / 'model_cache/BeatThis/final0.verified.ckpt', relay_host=None)


def original_state(args):
    progress, seals, leases = [], {}, {}
    for gpu, identity in enumerate(ORIGINAL_IDENTITIES):
        path = args.root / 'logs' / f'rollover_gpu{gpu}_progress.json'
        raw, seals[str(path)] = stable_read(path)
        record = base.strict_json(raw)
        if (not isinstance(record, dict) or record.get('worker') != WORKER
                or record.get('physical_gpu_index') != gpu
                or (record.get('pid'), record.get('start_ticks')) != identity
                or record.get('status') != 'running' or base.pid_alive(record)
                or not isinstance(record.get('run_id'), str) or not SAFE.fullmatch(record['run_id'])
                or record.get('lane_source_count') != 5000):
            raise PermissionError('original rollover PID/start/worker/scope is not dead and bound')
        old = record.get('old_consumed_ticket')
        if not isinstance(old, dict) or not isinstance(old.get('preserved_old_lease_sha256'), dict):
            raise PermissionError('original consumed-ticket evidence absent')
        old_progress = args.root / 'logs' / f'unified_gpu{gpu}_progress.json'
        oldraw, seals[str(old_progress)] = stable_read(old_progress)
        previous = base.strict_json(oldraw)
        if (old.get('previous_progress_path') != str(old_progress)
                or old.get('previous_progress_sha256') != seals[str(old_progress)]['sha256']
                or previous.get('worker') != WORKER or previous.get('physical_gpu_index') != gpu
                or previous.get('status') not in ('completed', 'failed_sealed_no_retry')
                or not previous.get('ended_unix') or base.pid_alive(previous)
                or any(old.get(k) != previous.get(k) for k in ('pid', 'start_ticks', 'assignment_sha256', 'status'))):
            raise PermissionError('consumed old terminal progress/evidence changed')
        expected = dict(old['preserved_old_lease_sha256'])
        newpath = args.root / 'gpu_locks' / f'physical_gpu{gpu}.newepoch.lease.json'
        paths = sorted((args.root / 'gpu_locks').glob(f'physical_gpu{gpu}*.lease.json'))
        if set(map(str, paths)) != set(expected) | {str(newpath)}:
            raise PermissionError('original physical lease path set changed/foreign')
        if str(args.root / 'gpu_locks' / f'physical_gpu{gpu}.lease.json') not in expected:
            raise PermissionError('original lease absent')
        matched_terminal = False
        for leasepath in paths:
            lraw, seals[str(leasepath)] = stable_read(leasepath)
            lease = base.strict_json(lraw)
            if (lease.get('worker') != WORKER or lease.get('physical_gpu_index') != gpu
                    or base.pid_alive(lease)):
                raise PermissionError('active/foreign original physical lease')
            if leasepath == newpath:
                if (any(lease.get(k) != record.get(k) for k in
                        ('pid', 'start_ticks', 'assignment_sha256', 'queue_sha256'))
                        or lease.get('edition') != base.EDITION or not lease.get('gpu_uuid')):
                    raise PermissionError('original rollover lease/progress source binding changed')
            else:
                if (seals[str(leasepath)]['sha256'] != expected[str(leasepath)]
                        or lease.get('assignment_sha256') != previous['assignment_sha256']):
                    raise PermissionError('preserved old lease hash/source binding changed')
                if (lease.get('pid'), lease.get('start_ticks')) == (previous['pid'], previous['start_ticks']):
                    if lease.get('status') != 'released_sealed_no_retry':
                        raise PermissionError('old terminal matching lease not sealed')
                    matched_terminal = True
            leases[str(leasepath)] = lease
        if not matched_terminal:
            raise PermissionError('old terminal owner lease evidence absent')
        lock = args.root / 'gpu_locks' / f'physical_gpu{gpu}.lock'
        if lock.resolve(strict=True) != lock or not stat.S_ISREG(lock.stat().st_mode):
            raise PermissionError('original physical lock missing/redirected')
        s = lock.stat()
        seals[str(lock)] = {'inode': s.st_ino, 'device': s.st_dev}
        progress.append(record)
    return progress, seals, leases


def assert_original_state(args):
    paths = set(str(p) for gpu in range(4)
                for p in (args.root / 'gpu_locks').glob(f'physical_gpu{gpu}*.lease.json'))
    if paths != set(args.original_leases):
        raise PermissionError('original lease path set changed/foreign')
    for path, seal in args.original_seals.items():
        if path.endswith('.lock'):
            p = Path(path)
            if p.resolve(strict=True) != p:
                raise PermissionError('original physical lock redirected')
            s = p.stat()
            if {'inode': s.st_ino, 'device': s.st_dev} != seal:
                raise PermissionError('original physical lock inode changed')
        else:
            raw = require_seal(path, seal)
            if path in args.original_leases and base.pid_alive(base.strict_json(raw)):
                raise PermissionError('original lease PID/start became active')
    for path, seal in args.executable_seals.items():
        require_seal(path, seal)
    for path, seal in args.metadata_seals.items():
        require_seal(path, seal)


def verify_disjoint_allocations(args, own_rows):
    ids, media, seals = set(), set(), {}
    for common, count, access_type in ((base.OLD_COMMON, 1000, old_transport.Access),
                                       (base.COMMON, 20000, transport.Access)):
        for worker in transport.WORKERS:
            access = access_type(worker)
            ticket_path = common / 'assignments' / (worker + '.ticket.json')
            manifest_path = common / 'assignments' / (worker + '.jsonl')
            traw, seals[str(ticket_path)] = stable_read(ticket_path)
            mraw, seals[str(manifest_path)] = stable_read(manifest_path)
            ticket = base.strict_json(traw)
            rows = [base.strict_json(line) for line in mraw.split(b'\n')[:-1]]
            if common == base.COMMON:
                qpath = common / (worker + '.queue.json')
                qraw, seals[str(qpath)] = stable_read(qpath)
                transport.validate_queue(worker, qraw, seals[str(ticket_path)]['sha256'])
            access.bind(ticket, rows, seals[str(ticket_path)]['sha256'], seals[str(manifest_path)]['sha256'])
            if len(rows) != count:
                raise PermissionError('original allocation count changed')
            for row in rows:
                key = (row['split'].casefold(), row['id'].casefold())
                paths = [row[k] for k in ('video_path', 'audio_path')]
                if key in ids or any(p in media for p in paths) or paths[0] == paths[1]:
                    raise PermissionError('original allocations intersect IDs/media; old4000 excluded')
                ids.add(key); media.update(paths)
            if common == base.COMMON and worker == WORKER and rows != own_rows:
                raise PermissionError('worker03 scope changed during disjoint verification')
    return seals


def recovery_owner(args, gpu):
    token = args.candidate / 'starts' / f'gpu{gpu}.json'
    lease = args.candidate / 'leases' / f'physical_gpu{gpu}.recovery.lease.json'
    if not os.path.lexists(token) and not os.path.lexists(lease):
        return None
    identity = base.strict_json(stable_read(token)[0])
    current = base.strict_json(stable_read(lease)[0]) if os.path.lexists(lease) else None
    required = {'pid', 'start_ticks', 'worker', 'physical_gpu_index', 'assignment_sha256',
                'edition', 'queue_sha256', 'lease_id', 'run_id', 'recovery_scope'}
    if (not isinstance(identity, dict) or set(identity) != required
            or (current is not None and any(current.get(k) != v for k, v in identity.items()))
            or identity['worker'] != WORKER or identity['physical_gpu_index'] != gpu
            or identity['assignment_sha256'] != args.assignment_sha
            or identity['queue_sha256'] != args.original_queue_sha or identity['edition'] != base.EDITION
            or identity['recovery_scope'] != 'unclaimed_original_indices_only'
            or type(identity['pid']) is not int or identity['pid'] <= 0
            or type(identity['start_ticks']) is not int or identity['start_ticks'] <= 0
            or not isinstance(identity['run_id'], str) or not SAFE.fullmatch(identity['run_id'])):
        raise PermissionError('recovery claim owner evidence malformed/foreign')
    return identity if current is not None else None


def validate_claim(record, index, row, args, recovery_owners):
    gpu = index % 4
    original = args.original_progress[gpu]
    owner = recovery_owners.get(gpu)
    expected = {'edition': base.EDITION, 'worker': WORKER, 'gpu': gpu,
        'assignment_sha256': args.assignment_sha, 'assignment_index': index,
        'source_row': row, 'split': row['split'], 'id': row['id'], 'automatic_retry': False}
    if (not isinstance(record, dict) or set(record) != CLAIM_KEYS
            or any(record.get(k) != v for k, v in expected.items())
            or type(record.get('gpu')) is not int or type(record.get('assignment_index')) is not int
            or type(record.get('pid')) is not int or type(record.get('owner_pid')) is not int
            or type(record.get('start_ticks')) is not int or record['automatic_retry'] is not False
            or type(record.get('claimed_unix')) not in (int, float) or record['claimed_unix'] <= 0):
        raise PermissionError('claim format/assignment/original row changed')
    candidates = [original] + ([owner] if owner is not None else [])
    if not any(record['pid'] == p['pid'] == record['owner_pid']
               and record['start_ticks'] == p['start_ticks'] and record['run_id'] == p['run_id']
               for p in candidates):
        raise PermissionError('claim original/recovery owner PID/start/run evidence mismatch')


def audit_claims(args, ticket, rows):
    owners = {g: recovery_owner(args, g) for g in range(4)}
    unclaimed, retained, counts = [], {}, [0] * 4
    root = Path(ticket['claims_root'])
    for index, row in enumerate(rows):
        path = root / row['split'] / (row['id'] + '.json')
        if os.path.lexists(path):
            raw, seal = stable_read(path, 1024 * 1024)
            validate_claim(base.strict_json(raw), index, row, args, owners)
            retained[str(path)] = seal
            counts[index % 4] += 1
        elif index % 4 == args.gpu:
            unclaimed.append((index, row))
    return unclaimed, retained, counts


def resource_budget(args, helper, run_root):
    local = args.root.resolve(strict=True)
    mounts = [line.split() for line in Path('/proc/mounts').read_text().splitlines()]
    filesystem = max((m for m in mounts if local.is_relative_to(Path(m[1]))), key=lambda m: len(m[1]))[2]
    if filesystem in {'ceph', 'nfs', 'nfs4', 'cifs', 'fuse.ceph', 'fuse.ceph-fuse', 'lustre'}:
        raise PermissionError('cache must be independent local disk')
    memory = helper.memory_contract(args.root)
    cached = base.tree_bytes(args.root / 'unified_runs' / base.EDITION)
    needed = 8 * base.GIB + max(0, 32 * base.GIB - cached)
    if memory['effective_headroom_bytes'] < 32 * base.GIB or memory['local_disk_free_bytes'] < needed:
        raise PermissionError('unchanged 32GiB headroom/four8GiB caches+8GiB reserve budget failed')
    if base.tree_bytes(run_root) > base.BUDGET:
        raise PermissionError('own retained cache exceeds original 8GiB budget')
    return {'memory_contract': memory, 'required_effective_headroom_bytes': 32 * base.GIB,
            'required_disk_free_bytes': needed}


def preflight(args):
    args.original_progress, args.original_seals, args.original_leases = original_state(args)
    args.executable_seals = {str(args.root / name): stable_read(args.root / name)[1] for name in SOURCES}
    for name, sha in SOURCES.items():
        if args.executable_seals[str(args.root / name)]['sha256'] != sha:
            raise PermissionError('frozen executable SHA changed')
    args.executable_seals[str(Path(__file__).resolve())] = stable_read(Path(__file__).resolve())[1]
    access = RecoveryAccess(WORKER)
    queue = base.verify_queue(args, access)
    args.original_queue, args.original_queue_sha = queue, args.queue_sha
    ticket, rows, sha, signatures, helper = base.verify_assignment(args, access, queue)
    args.assignment_sha = sha
    for ref in ticket['code_refs']:
        path = args.code_root / Path(ref['path']).name
        args.executable_seals[str(path)] = stable_read(path)[1]
        if args.executable_seals[str(path)]['sha256'] != ref['sha256']:
            raise PermissionError('core source changed after frozen assignment verification')
    helper_path = args.root / 'fullvideo_validated_helpers_ed283d684.py'
    args.executable_seals[str(helper_path)] = stable_read(helper_path)[1]
    if args.executable_seals[str(helper_path)]['sha256'] != base.HELPER_SHA:
        raise PermissionError('memory helper changed after verification')
    for record in args.original_progress:
        if (record.get('assignment_sha256') != sha or record.get('queue_sha256') != args.queue_sha
                or record.get('queue_path') != args.queue or record.get('code_signatures') != signatures):
            raise PermissionError('original rollover assignment/code/queue differs from frozen evidence')
    oldraw = require_seal(args.previous_progress, args.original_seals[str(args.previous_progress)])
    previous = base.strict_json(oldraw)
    old_evidence = args.original_progress[args.gpu]['old_consumed_ticket']
    evidence = {'previous_progress_path': str(args.previous_progress),
                'previous_progress_sha256': base.digest(oldraw),
                'preserved_old_lease_sha256': dict(old_evidence['preserved_old_lease_sha256'])}
    evidence.update(base.verify_models(args))
    evidence = base.verify_previous(args, access, queue, previous, evidence, ticket, rows, signatures)
    if evidence != old_evidence:
        raise PermissionError('real old consumed evidence differs from original rollover proof')
    args.previous_evidence = evidence
    args.metadata_seals = verify_disjoint_allocations(args, rows)
    args.unclaimed, args.retained_claims, args.claim_counts = audit_claims(args, ticket, rows)
    if os.path.lexists(args.candidate / 'starts' / f'gpu{args.gpu}.json'):
        raise PermissionError('recovery attempt already sealed; no automatic retry')
    if os.path.lexists(args.candidate / 'leases' / f'physical_gpu{args.gpu}.recovery.lease.json'):
        raise PermissionError('existing recovery lease retained; no retry')
    if os.path.lexists(args.candidate / 'logs' / f'gpu{args.gpu}' / 'progress.json'):
        raise PermissionError('existing recovery progress retained; no retry')
    gate = RecoveryGate(args, sha, helper, args.candidate / 'runs')
    gpu = gate.gpu(True)
    old_uuid = args.original_leases[str(args.root / 'gpu_locks' / f'physical_gpu{args.gpu}.newepoch.lease.json')]['gpu_uuid']
    if gpu['gpu_uuid'] != old_uuid:
        raise PermissionError('fresh physical GPU UUID differs from original lease')
    args.expected_gpu_uuid = old_uuid
    resource = resource_budget(args, helper, args.candidate / 'runs')
    assert_original_state(args)
    access.recovery_args = args
    return access, ticket, rows, sha, signatures, helper, {**gpu, **resource}


class RecoveryAccess(transport.Access):
    def claim(self, path, record):
        created = super().claim(path, record)
        if not created:
            args = self.recovery_args
            owners = {g: recovery_owner(args, g) for g in range(4)}
            existing = base.strict_json(stable_read(Path(path), 1024 * 1024)[0])
            validate_claim(existing, record['assignment_index'], record['source_row'], args, owners)
        return created


class RecoveryGate(base.Gate):
    def __init__(self, args, assignment_sha, helper, run_root):
        super().__init__(args, assignment_sha, helper, run_root)
        self.lease = args.candidate / 'leases' / f'physical_gpu{args.gpu}.recovery.lease.json'
        self.token = args.candidate / 'starts' / f'gpu{args.gpu}.json'
        self.identity.update(run_id=getattr(args, 'recovery_run_id', 'check-only'),
                             recovery_scope='unclaimed_original_indices_only')

    def check(self, fresh=False):
        with self.mutex:
            assert_original_state(self.args)
            token = base.strict_json(require_seal(self.token, self.token_seal))
            if token != self.identity:
                raise PermissionError('recovery startup identity changed')
            paths = list((self.args.candidate / 'leases').glob(f'physical_gpu{self.args.gpu}*.lease.json'))
            if paths != [self.lease]:
                raise PermissionError('foreign recovery lease path; fail closed')
            if self.lease.resolve(strict=True) != self.lease:
                raise PermissionError('recovery lease redirected')
            require_seal(self.lease, self.lease_seal)
            result = super().check(fresh)
            self.lease_seal = stable_read(self.lease)[1]
            if result['gpu_uuid'] != self.args.expected_gpu_uuid:
                raise PermissionError('recovery physical UUID changed')
            return result

    def __enter__(self):
        # Reserve the sole attempt before lock/resource admission; failures keep the token.
        transport.anchor(self.token)
        transport.anchor(self.lease)
        transport.anchor(self.run_root)
        self.token.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.token, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(base.raw_json(self.identity)); stream.flush(); os.fsync(stream.fileno())
        transport.sync(self.token.parent)
        self.token_seal = stable_read(self.token)[1]
        try:
            assert_original_state(self.args)
            # Never create/replace the existing physical lock inode.
            self.fd = os.open(self.lock_dir / f'physical_gpu{self.args.gpu}.lock', os.O_RDWR | os.O_NOFOLLOW)
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert_original_state(self.args)
            if self.gpu(True)['gpu_uuid'] != self.args.expected_gpu_uuid:
                raise PermissionError('startup fresh UUID changed')
            resource_budget(self.args, self.helper, self.run_root)
            if base.verify_models(self.args) != {k: self.args.previous_evidence[k] for k in
                    ('model_signatures', 'old_cache_identity_sha256', 'old_verified_models_unchanged')}:
                raise PermissionError('verified models changed before startup')
            for path in self.args.retained_claims:
                require_seal(path, self.args.retained_claims[path])
            self.lease.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            self.created = True
            with os.fdopen(fd, 'wb') as stream:
                stream.write(base.raw_json(self.identity)); stream.flush(); os.fsync(stream.fileno())
            transport.sync(self.lease.parent)
            self.lease_seal = stable_read(self.lease)[1]
            self.uuid = self.args.expected_gpu_uuid
            self.check(True)
            return self
        except BaseException as error:
            try:
                if self.created:
                    self.seal_lease(error)
            finally:
                if self.fd is not None:
                    os.close(self.fd); self.fd = None
            raise

    def seal_lease(self, error=None):
        require_seal(self.lease, self.lease_seal)
        base.atomic_json(self.lease, {**self.identity, 'gpu_uuid': self.uuid,
            'status': 'released_sealed_no_retry', 'ended_unix': time.time(),
            **({'error': str(error)} if error is not None else {})})
        self.lease_seal = stable_read(self.lease)[1]

    def __exit__(self, *exc):
        if self.fd is not None:
            with self.mutex:
                try:
                    self.seal_lease(exc[1] if len(exc) > 1 else None)
                finally:
                    os.close(self.fd); self.fd = None


class RecoveryLane(base.Lane):
    def __init__(self, args, access, ticket, rows, assignment_sha, signatures, helper):
        super().__init__(args, access, ticket, rows, assignment_sha, signatures, helper)
        self.rows = list(args.unclaimed)
        self.run_id = f'recovery03-g{args.gpu}-{time.time_ns()}-{uuid.uuid4().hex[:8]}'
        args.recovery_run_id = self.run_id
        self.edition_root = args.candidate / 'runs'
        self.work = self.edition_root / self.run_id
        self.progress_path = args.candidate / 'logs' / f'gpu{args.gpu}' / 'progress.json'
        self.progress.update(run_id=self.run_id, lane_source_count=len(self.rows),
            counts_scope='this_recovery_increment_only_not_total80k', original_lane_source_count=5000,
            original_claims_retained_and_excluded=args.claim_counts[args.gpu],
            original_progress_path=str(args.root / 'logs' / f'rollover_gpu{args.gpu}_progress.json'),
            original_progress_preserved=True, recovery_scope='unclaimed_original_indices_only',
            publication_only_recovery=False, transfers_enabled=False, automatic_retry=False)

    def report(self, **values):
        transport.anchor(self.progress_path)
        base.Lane.report(self, **values)

    def refresh_queue(self):
        base.Lane.refresh_queue(self)
        if self.args.queue_sha != self.args.original_queue_sha:
            raise PermissionError('original queue bytes changed; no replacement queue')

    def run(self):
        # Frozen Lane.run orchestration, with only the Gate factory replaced.
        transport.anchor(self.work)
        transport.anchor(self.progress_path)
        self.work.mkdir(parents=True, exist_ok=False)
        self.gate = RecoveryGate(self.args, self.assignment_sha, self.helper, self.work)
        with self.gate:
            sys.path.insert(0, str(self.args.code_root))
            self.core = base.load_module('unified_algorithm_core', self.args.code_root / 'ms_unified_novlm_v1.py')
            self.core.write_once(self.work / 'resource_evidence.json', base.raw_json(self.stage()))
            self.report(status='models_loading', code_signatures=self.signatures)
            monitor = threading.Thread(target=self.heartbeat, daemon=True); monitor.start()
            try:
                started = time.perf_counter()
                self.models = self.core.load_models({'gpu_index': self.args.gpu,
                    'scail_root': str(self.args.scail_root), 'beat_checkpoint': str(self.args.beat_checkpoint)})
                self.progress['profile_seconds']['models_load_once'] = time.perf_counter() - started
                with ThreadPoolExecutor(max_workers=1) as executor:
                    while self.next < len(self.rows) or self.pending:
                        self.stage()
                        while self.enqueue(len(self.rows), executor):
                            pass
                        if self.pending:
                            self.process()
                        if self.receipts and (len(self.receipts) >= 20 or not self.pending):
                            self.publish(False)
                            self.refresh_queue()
                    if self.receipts:
                        self.publish(False)
                self.report(status='completed', stage='queue_exhausted', ended_unix=time.time(),
                            requests_more_authorized_sources=True, no_self_selected_sources=True)
            except BaseException as error:
                self.report(status='failed_sealed_no_retry', error=str(error), ended_unix=time.time())
                raise
            finally:
                self.stop.set(); monitor.join()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True, choices=range(4))
    parser.add_argument('--check-only', action='store_true')
    options = parser.parse_args(argv)
    if not options.check_only:
        parser.error('draft-only: durable old-consumer retirement and recovery/transfer exclusion seal absent; no real launch')
    args = fixed_args(options.gpu)
    access, ticket, rows, sha, signatures, helper, resource = preflight(args)
    if options.check_only:
        print(json.dumps({'status': 'verified_no_models_claims_locks', 'worker': WORKER, 'gpu': args.gpu,
            'original_scope': len(rows), 'original_lane_scope': len(rows) // 4,
            'retained_claims_all_scope': sum(args.claim_counts), 'retained_claims_lane': args.claim_counts[args.gpu],
            'unclaimed_original_lane_indices': len(args.unclaimed), 'fresh_gpu_memory_mib': resource['memory_used_mib'],
            'old4000_and_other3worker_intersections': 0, 'original_progress_preserved': True,
            'required_memory_headroom_gib': resource['required_effective_headroom_bytes'] // base.GIB,
            'required_disk_free_bytes': resource['required_disk_free_bytes'], 'physical_lock_acquired': False, 'launch_admitted': False,
            'durable_old_consumer_retirement_verified': False,
            'recovery_transfer_exclusion_seal_verified': False}), flush=True)
        return 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    lane = RecoveryLane(args, access, ticket, rows, sha, signatures, helper)
    try:
        lane.run()
    except BaseException as error:
        if getattr(lane, 'gate', None) is not None and lane.gate.created and not lane.progress_path.exists():
            lane.report(status='failed_sealed_no_retry', stage='resource_admission_failed',
                        error=str(error), ended_unix=time.time())
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
