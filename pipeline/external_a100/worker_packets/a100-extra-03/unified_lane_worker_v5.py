"""Finite new-edition GPU lane: real local lease; unchanged unified algorithm."""
from __future__ import annotations

import argparse, fcntl, hashlib, importlib.util, io, json, os, re, resource
import shutil, subprocess, sys, tarfile, tempfile, threading, time, uuid
import xml.etree.ElementTree as ET
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GIB = 2 ** 30
BUDGET, OUTPUT_CAP, INPUT_CAP = 8 * GIB, 4 * GIB, 2 * GIB
COMMON = Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/external_ms_unified_run_20261027_v1')
EDITION = COMMON.name
HELPER_SHA = 'ed283d6844830a1c2da109361391dbc44858a16c0e0262ca3ace117b3c48cb78'
sys.dont_write_bytecode = True

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while raw := stream.read(1024 * 1024):
            h.update(raw)
    return h.hexdigest()

def raw_json(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()

def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.own-progress-', delete=False) as f:
        temporary = Path(f.name)
        f.write(raw_json(value)); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_DIRECTORY | os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def tree_bytes(path):
    total = 0
    for p in path.rglob('*'):
        try:
            if p.is_file():
                total += p.stat().st_size
        except FileNotFoundError:
            pass  # Own committed cache can be reclaimed while heartbeat samples.
    return total

def start_ticks(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def verify_assignment(args, access):
    ticket_raw, assignment_raw = access.read(args.ticket), access.read(args.manifest)
    ticket, assignment_sha = json.loads(ticket_raw), digest(assignment_raw)
    if (ticket.get('schema') != 'ms_external_unified_assignment_v1'
            or ticket.get('worker_id') != args.worker or ticket.get('edition') != EDITION
            or ticket.get('source_count') != 1000 or ticket.get('assignment_sha256') != assignment_sha
            or ticket.get('allowed_gpu_indices') != [0, 1, 2, 3]
            or Path(args.ticket) != COMMON / 'assignments' / (args.worker + '.ticket.json')
            or Path(args.manifest) != COMMON / 'assignments' / (args.worker + '.jsonl')
            or ticket.get('assignment_path') != args.manifest
            or Path(ticket.get('output_root', '')) != COMMON / 'results' / args.worker
            or Path(ticket.get('claims_root', '')) != COMMON / 'claims'):
        raise ValueError('original own-worker ticket/assignment contract mismatch')
    if not assignment_raw or bytes([0]) in assignment_raw or not assignment_raw.endswith(bytes([10])):
        raise ValueError('NUL or partial assignment; cursor retained at zero')
    rows = [json.loads(line) for line in assignment_raw.split(bytes([10]))[:-1]]
    seen = set()
    for row in rows:
        if set(row) != {'id', 'split', 'video_path', 'audio_path', 'meta_path', 'source_ordinal'}:
            raise ValueError('original six-key manifest row required')
        key = (row['split'], row['id'])
        if key in seen or any(re.fullmatch(r'[A-Za-z0-9_-]+', row[k]) is None for k in ('split', 'id')):
            raise ValueError('unsafe or duplicate split/id')
        seen.add(key)
        for k in ('video_path', 'audio_path', 'meta_path'):
            if not Path(row[k]).is_absolute():
                raise ValueError('original absolute input path required')
        if any(Path(row[k]).stem != row['id'] for k in ('video_path', 'audio_path')):
            raise ValueError('media source ID mismatch')
        if type(row['source_ordinal']) is not int:
            raise ValueError('source ordinal must remain original integer')
    if len(rows) != 1000:
        raise ValueError('exactly 1000 original source rows required')
    refs = ticket.get('code_refs', [])
    expected = {'ms_unified_novlm_v1.py', 'ms_fullframe_pose2d_draft_v2.py',
                'ms_fullframe_pose2d_draft_v1.py', 'ms_pose2d_draft.py', 'dwpose_full_extract.py'}
    if len(refs) != 5 or {Path(r['path']).name for r in refs} != expected:
        raise ValueError('exact five algorithm references required')
    signatures = {Path(r['path']).name: file_sha(args.code_root / Path(r['path']).name) for r in refs}
    if any(signatures[Path(r['path']).name] != r['sha256'] for r in refs):
        raise ValueError('actual local snapshot SHA mismatch')
    helper = args.root / 'fullvideo_validated_helpers_ed283d684.py'
    if file_sha(helper) != HELPER_SHA:
        raise ValueError('verified independent memory helper changed')
    access.bind(ticket, rows, digest(ticket_raw), assignment_sha)
    return ticket, rows, assignment_sha, signatures, load_module('unified_memory_helper', helper)

class Gate:
    def __init__(self, args, assignment_sha, helper, run_root):
        self.args, self.helper, self.run_root = args, helper, run_root
        self.lock_dir = args.root / 'gpu_locks'
        self.lease = self.lock_dir / f'physical_gpu{args.gpu}.lease.json'
        self.identity = {'pid': os.getpid(), 'start_ticks': start_ticks(os.getpid()),
                         'worker': args.worker, 'physical_gpu_index': args.gpu,
                         'assignment_sha256': assignment_sha, 'lease_id': uuid.uuid4().hex}
        self.fd, self.uuid, self.peak = None, None, 0
        if getattr(args, 'resume_first3', False):
            previous = json.loads(self.lease.read_bytes())
            if previous.get('status') != 'released_sealed_no_retry' or previous.get('worker') != args.worker or previous.get('assignment_sha256') != assignment_sha or Path('/proc/' + str(previous['pid'])).exists():
                raise PermissionError('previous sealed lease not safely ended; unchanged')
            self.identity['prior_sealed_lease_path'] = str(self.lease)
            self.lease = self.lock_dir / ('physical_gpu0.continuation_' + self.identity['lease_id'] + '.lease.json')
        elif getattr(args, 'recover_preflight', False):
            if args.worker != 'a100-extra-03':
                raise PermissionError('only03 explicit preflight-only continuation')
            previous = json.loads(self.lease.read_bytes())
            proof = json.loads((args.root / 'logs/03_preflight_no_source_claims.json').read_bytes())
            if previous['pid'] not in proof['refused_pids_without_source_claims'] or Path('/proc/' + str(previous['pid'])).exists() or previous.get('assignment_sha256') != assignment_sha:
                raise PermissionError('preflight proof does not authorize untouched sources')
            self.identity['prior_preflight_lease_path'] = str(self.lease)
            self.lease = self.lock_dir / ('physical_gpu' + str(args.gpu) + '.preflight_continuation_' + self.identity['lease_id'] + '.lease.json')
        self.mutex = threading.RLock()

    def gpu_once(self, fresh):
        result = subprocess.run(['nvidia-smi', '-i', str(self.args.gpu), '-q', '-x'],
                                check=True, capture_output=True, text=True, timeout=20)
        nodes = ET.fromstring(result.stdout).findall('gpu')
        if len(nodes) != 1:
            raise PermissionError('physical GPU query ambiguous')
        node = nodes[0]
        uuid_value = node.findtext('uuid')
        index = subprocess.run(['nvidia-smi', '-i', str(self.args.gpu), '--query-gpu=index,uuid',
                                '--format=csv,noheader,nounits'], check=True,
                               capture_output=True, text=True, timeout=20).stdout.strip().split(',')
        if len(index) != 2 or int(index[0]) != self.args.gpu or index[1].strip() != uuid_value:
            raise PermissionError('physical GPU index/UUID verification failed')
        pids = [int(p.findtext('pid')) for p in node.findall('processes/process_info')]
        used_text = node.findtext('fb_memory_usage/used', '')
        if not re.fullmatch(r'[0-9]+ MiB', used_text):
            raise PermissionError('GPU memory reading not verifiable')
        used = int(used_text.split()[0])
        if any(pid != os.getpid() for pid in pids) or (fresh and pids):
            raise PermissionError('GPU compute/graphics PID occupied; unchanged')
        if fresh and used != 0:
            raise PermissionError('fresh_memory_not_zero_without_process; unchanged')
        if self.uuid is not None and uuid_value != self.uuid:
            raise PermissionError('physical GPU UUID changed')
        self.peak = max(self.peak, used)
        return {'gpu_uuid': uuid_value, 'memory_used_mib': used, 'process_pids': pids,
                'foreign_compute_or_graphics_pids': [], 'physical_gpu_index': self.args.gpu}

    def gpu(self, fresh):
        deadline = time.monotonic() + 30
        while True:
            try:
                return self.gpu_once(fresh)
            except PermissionError as error:
                if not fresh or str(error) != 'fresh_memory_not_zero_without_process; unchanged' or time.monotonic() >= deadline:
                    raise
                time.sleep(1)

    def check(self, fresh=False):
        with self.mutex:
            stat = os.fstat(self.fd)
            lock = self.lock_dir / f'physical_gpu{self.args.gpu}.lock'
            if (stat.st_ino, stat.st_dev) != (lock.stat().st_ino, lock.stat().st_dev):
                raise PermissionError('held lock FD no longer names physical GPU lock')
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            current = json.loads(self.lease.read_bytes())
            if any(current.get(k) != v for k, v in self.identity.items()):
                raise PermissionError('lease is not owned by this PID/start/assignment')
            local = self.args.root.resolve(strict=True)
            mounts = [line.split() for line in Path('/proc/mounts').read_text().splitlines()]
            filesystem = max((m for m in mounts if local.is_relative_to(Path(m[1]))), key=lambda m: len(m[1]))[2]
            if filesystem in {'ceph', 'nfs', 'nfs4', 'cifs', 'fuse.ceph', 'fuse.ceph-fuse', 'lustre'}:
                raise PermissionError('cache must be independent local disk')
            memory = self.helper.memory_contract(self.args.root)
            if memory['effective_headroom_bytes'] < 32 * GIB:
                raise PermissionError('actual effective memory headroom below 32GiB')
            cached = tree_bytes(self.args.root / 'unified_runs' / EDITION)
            needed = 8 * GIB + max(0, 32 * GIB - cached)
            if memory['local_disk_free_bytes'] < needed:
                raise PermissionError('disk cannot cover four 8GiB caches plus 8GiB reserve')
            if tree_bytes(self.run_root) > BUDGET:
                raise PermissionError('own retained input/output/failed cache exceeds 8GiB')
            gpu = self.gpu(fresh)
            atomic_json(self.lease, {**self.identity, 'gpu_uuid': gpu['gpu_uuid'], 'heartbeat_unix': time.time()})
            return {**self.identity, **gpu, 'memory_contract': memory,
                    'required_effective_headroom_bytes': 32 * GIB, 'required_disk_free_bytes': needed,
                    'actual_lock_fd': self.fd, 'observed_unix': time.time()}

    def __enter__(self):
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.lock_dir / f'physical_gpu{self.args.gpu}.lock',
                          os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.lease.exists():
                raise PermissionError('existing lease retained; no overwrite or automatic retry')
            fd = os.open(self.lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'wb') as f:
                f.write(raw_json(self.identity)); f.flush(); os.fsync(f.fileno())
            evidence = self.check(True)
            self.uuid = evidence['gpu_uuid']
            return self
        except BaseException:
            os.close(self.fd); self.fd = None
            raise

    def __exit__(self, *exc):
        if self.fd is not None:
            with self.mutex:
                atomic_json(self.lease, {**self.identity, 'gpu_uuid': self.uuid,
                                        'status': 'released_sealed_no_retry', 'ended_unix': time.time()})
                os.close(self.fd); self.fd = None

class Lane:
    def __init__(self, args, access, ticket, rows, assignment_sha, signatures, helper):
        self.args, self.access, self.ticket = args, access, ticket
        self.assignment_sha, self.signatures, self.helper = assignment_sha, signatures, helper
        self.original_rows = rows
        self.rows = [(i, r) for i, r in enumerate(rows) if i % 4 == args.gpu]
        assert len(self.rows) == 250
        self.run_id = f'g{args.gpu}-{time.time_ns()}-{uuid.uuid4().hex[:8]}'
        self.edition_root = args.root / 'unified_runs' / EDITION
        self.work = self.edition_root / self.run_id
        self.marker = self.edition_root / 'first3_verified.json'
        self.shared = Path(ticket['output_root'])
        self.pending, self.receipts, self.inputs = deque(), [], []
        self.next, self.batch_no, self.reserved, self.output_used = 0, 0, 0, 0
        self.stop, self.fatal = threading.Event(), None
        self.progress_path = args.root / 'logs' / f'unified_gpu{args.gpu}_progress.json'
        self.progress = {'worker': args.worker, 'physical_gpu_index': args.gpu,
                         'assignment_sha256': assignment_sha, 'run_id': self.run_id,
                         'pid': os.getpid(), 'start_ticks': start_ticks(os.getpid()),
                         'started_unix': time.time(), 'unique_processed': 0, 'published': 0,
                         'published_generated_with_events': 0,
                         'success': 0, 'partial': 0, 'needs_regeneration': 0, 'failed': 0,
                         'existing_claims': 0, 'frame_count': 0, 'profile_seconds': {}}

    def report(self, **values):
        self.progress.update(values, heartbeat_unix=time.time(),
                             rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                             gpu_memory_peak_mib=self.gate.peak, backlog=len(self.rows) - self.next,
                             pending_sources=len(self.pending), cache_reserved_bytes=self.reserved,
                             cache_actual_bytes=tree_bytes(self.work))
        atomic_json(self.progress_path, {**self.progress, 'profile_seconds': dict(self.progress['profile_seconds'])})

    def heartbeat(self):
        while not self.stop.wait(60):
            try:
                self.gate.check(); self.report()
            except BaseException as error:
                self.fatal = error; self.stop.set()

    def stage(self):
        if self.fatal is not None:
            raise RuntimeError('heartbeat gate failed') from self.fatal
        return self.gate.check()

    def batch_paths(self):
        name = f'{self.run_id}-batch{self.batch_no:03d}'
        return self.work / name, self.shared / 'batches' / f'gpu{self.args.gpu}' / name

    def verify_first3(self, marker):
        expected = {'worker': self.args.worker, 'assignment_sha256': self.assignment_sha,
                    'code_signatures': self.signatures, 'edition': EDITION}
        if any(marker.get(k) != v for k, v in expected.items()) or marker.get('indices') != [0, 4, 8]:
            raise PermissionError('own current-edition first3 marker does not match original assignment/signatures')
        target = Path(marker['commit_path'])
        if not target.is_relative_to(self.shared / 'batches' / 'gpu0'):
            raise PermissionError('foreign-node or historical first3 acceptance forbidden')
        raw = self.access.read(str(target), expected_sha=marker['commit_sha256'])
        if digest(raw) != marker['commit_sha256']:
            raise PermissionError('published first3 commit SHA mismatch')
        commit = json.loads(raw)
        if (commit.get('assignment_sha256') != self.assignment_sha or commit.get('worker') != self.args.worker
                or commit.get('assignment_indices') != [0, 4, 8] or commit.get('validation_first3') is not True
                or commit.get('unique_sources_attempted') != 3 or len(commit.get('receipts', [])) != 3):
            raise PermissionError('published first3 commit identity mismatch')
        payloads = {}
        for item in commit['publications']:
            payload = self.access.read(item['path'], expected_sha=item['sha256'])
            if digest(payload) != item['sha256']:
                raise PermissionError('published first3 file SHA mismatch')
            payloads[item['path']] = payload
        sys.path.insert(0, str(self.args.code_root))
        rules = load_module('unified_first3_rules', self.args.code_root / 'ms_fullframe_pose2d_draft_v2.py')
        import numpy as np
        for index in (0, 4, 8):
            row = self.rows[index // 4][1] if self.args.gpu == 0 else self.original_rows[index]
            case = target.parent / 'cases' / row['split'] / row['id']
            manifest = json.loads(payloads[str(case / 'fullframe_manifest.json')])
            draft = json.loads(payloads[str(case / 'draft.json')])
            beats = json.loads(payloads[str(case / 'beats.json')])
            receipt = json.loads(payloads[str(case / 'receipt.json')])
            pose_raw = payloads[str(case / 'pose.npz')]
            if digest(pose_raw) != manifest['npz_sha256'] or receipt['source_row'] != row:
                raise PermissionError('first3 actual NPZ/source receipt identity mismatch')
            bound_row = {'id': row['id'], 'split': row['split'], 'npz_path': str(case / 'pose.npz'),
                         'fullframe_manifest_path': str(case / 'fullframe_manifest.json')}
            with np.load(io.BytesIO(pose_raw), allow_pickle=False) as archive:
                data = {k: archive[k] for k in archive.files}
            mask = rules.validate_observations(bound_row, manifest, data)
            if rules.fullframe_binding(manifest, data) != draft['source']['fullframe_binding']:
                raise PermissionError('first3 draft binding does not match actual observations')
            if (not draft['events'] or not draft['ms_text'].strip()
                    or beats.get('audio_path') != row['audio_path'] or beats.get('device') != 'cuda:0'
                    or beats.get('float16') is not False
                    or any(draft.get(k) is not v or receipt.get(k) is not v for k, v in rules.FLAGS.items())):
                raise PermissionError('first3 actions/BeatThis/unreviewed flags not verified')
            for event in draft['events']:
                frames, joints = event['support']['frame_indices'], event['support']['joint_indices']
                if not event['text'].strip() or not frames or not mask[np.ix_(frames, joints)].all():
                    raise PermissionError('first3 action support lacks usable native observations')
        return marker

    def enqueue(self, boundary, executor):
        if self.next >= boundary or len(self.pending) >= 2 or len(self.receipts) + len(self.pending) >= (3 if boundary == 3 else 20):
            return False
        index, row = self.rows[self.next]
        size = sum(self.access.stat(row[k])['bytes'] for k in ('video_path', 'audio_path'))
        if size > INPUT_CAP:
            raise ValueError('input exceeds 2GiB per-source cap; stop without claim/retry')
        if tree_bytes(self.work) + self.reserved + size + 2 * self.output_used + 64 * 1024 ** 2 > BUDGET:
            if self.receipts:
                return False
            raise ValueError('retained failed cache prevents bounded copy; stop without retry')
        record = {'edition': EDITION, 'worker': self.args.worker, 'gpu': self.args.gpu,
                  'assignment_sha256': self.assignment_sha, 'assignment_index': index,
                  'source_row': row, 'split': row['split'], 'id': row['id'], 'pid': os.getpid(),
                  'owner_pid': os.getpid(), 'start_ticks': start_ticks(os.getpid()),
                  'claimed_unix': time.time(), 'automatic_retry': False, 'run_id': self.run_id}
        claimed = self.access.claim(str(Path(self.ticket['claims_root']) / row['split'] / (row['id'] + '.json')), record)
        self.next += 1
        if not claimed:
            self.receipts.append({**record, 'status': 'existing_claim_not_reassigned'})
            self.progress['existing_claims'] += 1
            if boundary == 3:
                raise PermissionError('first3 already claimed; never reuse or retry')
            return True
        profile, started = {}, time.perf_counter()
        self.reserved += size
        copier = self.access.copy if self.args.relay_host else self.core.copy_source
        future = executor.submit(copier, row, self.work / 'staging', INPUT_CAP, profile)
        self.pending.append((index, row, future, profile, started, size))
        return True

    def process(self):
        index, row, future, profile, started, size = self.pending.popleft()
        local_batch, target = self.batch_paths()
        case = local_batch / 'cases' / row['split'] / row['id']
        input_dir = self.work / 'staging' / row['split'] / row['id']
        self.report(status='running', current_source=row['id'], assignment_index=index, stage='wait_copy')
        wait = time.perf_counter()
        try:
            try:
                copied, hashes = future.result()
            finally:
                self.reserved -= size
                profile['wait_copy'] = time.perf_counter() - wait
            self.stage()
            remaining = min(OUTPUT_CAP - self.output_used, (BUDGET - tree_bytes(self.work) - self.reserved) // 3)
            def probe(path):
                actual = self.core.probe_video(path)
                n = len(actual.get('frames', []))
                if n * 4096 + len(raw_json(actual)) * 3 > remaining:
                    raise ValueError('native-array/output preallocation exceeds remaining cache budget')
                self.progress['current_total_frames'] = n
                return actual
            result = self.core.extract_and_rule(row, copied, hashes, self.models, case,
                       target / 'cases' / row['split'] / row['id'], probe=probe,
                       max_output_bytes=remaining, profile=profile, source_started=started)
            draft = json.loads((case / 'draft.json').read_bytes())
            result['event_count'] = len(draft['events'])
            result['assignment_index'] = index
            result['copy_evidence'] = self.access.copy_evidence.get((row['split'], row['id'])) or {
                'mode': 'direct_core_copy_source_one_pass', 'input_sha256': hashes,
                'original_paths': {k: row[k] for k in copied}, 'copied_bytes': size}
            result['generated_with_action_events'] = bool(draft['events'])
            atomic_json(case / 'receipt.json', result)
            if self.args.gpu == 0 and self.next <= 3:
                if not draft['events'] or not draft['ms_text'].strip() or any(
                        not e['text'].strip() or not e['support']['frame_indices'] for e in draft['events']):
                    raise ValueError('first3 requires real nonempty action events/text/support frames')
                if any(draft.get(k) is not v for k, v in self.core.v2.FLAGS.items()):
                    raise ValueError('first3 unreviewed diagnostic flags mismatch')
            self.inputs.append(input_dir)
            self.progress['frame_count'] += result['frame_count']
            status = result['draft_status']
            self.progress['success' if status == 'draft_generated' else status] += 1
        except Exception as error:
            failed = self.work / 'failed' / row['split'] / row['id']
            failed.mkdir(parents=True, exist_ok=True)
            if case.exists():
                case.rename(failed / 'partial_outputs')
            case.mkdir(parents=True, exist_ok=True)
            result = {'schema': 'ms_unified_novlm_case_receipt_v1', 'source_row': row,
                      'assignment_index': index, 'status': 'failed_sealed_no_retry',
                      'error': str(error), 'profile_seconds': profile, **self.core.v2.FLAGS}
            self.core.write_once(case / 'receipt.json', raw_json(result))
            self.progress['failed'] += 1
        self.output_used = tree_bytes(local_batch)
        self.receipts.append(result); self.progress['unique_processed'] += 1
        for key, value in profile.items():
            totals = self.progress['profile_seconds']; totals[key] = totals.get(key, 0) + value
        self.report(status='running', stage='source_complete')
        if self.output_used > OUTPUT_CAP or tree_bytes(self.work) + self.reserved + self.output_used > BUDGET:
            raise ValueError('cache boundary exceeded; retained sealed outputs, no expansion')
        if self.args.gpu == 0 and self.next <= 3 and result['status'] == 'failed_sealed_no_retry':
            self.publish(False)
            raise RuntimeError('first3 source failed; no expansion or retry')

    def publish(self, validation):
        self.stage()
        local_batch, target = self.batch_paths()
        local_batch.mkdir(parents=True, exist_ok=True)
        archive = local_batch / 'batch.tar'
        if tree_bytes(self.work) + self.reserved + self.output_used + 1024 ** 2 > BUDGET:
            raise ValueError('archive duplication would exceed 8GiB; preserve unpublished cache')
        publications = []
        started = time.perf_counter()
        with tarfile.open(archive, 'x') as pack:
            for case in sorted((local_batch / 'cases').glob('*/*')):
                for name in self.core.FILES:
                    path = case / name
                    if path.is_file():
                        relative = path.relative_to(local_batch)
                        pack.add(path, arcname=str(relative), recursive=False)
                        sha = file_sha(path)
                        receipt = self.access.publish_file(str(path), str(target / relative), sha)
                        publications.append({'path': str(target / relative), 'sha256': sha, 'receipt': receipt})
        with archive.open('rb') as f:
            os.fsync(f.fileno())
        archive_sha = file_sha(archive)
        archive_receipt = self.access.publish_file(str(archive), str(target / 'batch.tar'), archive_sha)
        publications.append({'path': str(target / 'batch.tar'), 'sha256': archive_sha, 'receipt': archive_receipt})
        self.stage()
        commit = {'schema': 'ms_unified_novlm_batch_commit_v1', 'edition': EDITION,
                  'worker': self.args.worker, 'gpu': self.args.gpu, 'run_id': self.run_id,
                  'assignment_sha256': self.assignment_sha, 'code_signatures': self.signatures,
                  'assignment_indices': [r.get('assignment_index') for r in self.receipts],
                  'receipts': self.receipts, 'batch_sha256': archive_sha, 'publications': publications,
                  'validation_first3': validation, 'quality_approved': False,
                  'unique_sources_attempted': sum(r['status'] != 'existing_claim_not_reassigned' for r in self.receipts),
                  'unique_sources_with_action_events': sum(bool(r.get('generated_with_action_events')) for r in self.receipts),
                  'frame_count': sum(r.get('frame_count', 0) for r in self.receipts),
                  'profile_seconds': {'pack_publish': time.perf_counter() - started}, **self.core.v2.FLAGS}
        self.access.publish_json(commit, str(target / 'commit.json'))
        commit_raw = self.access.read(str(target / 'commit.json'), expected_sha=digest(raw_json(commit)))
        if json.loads(commit_raw) != commit:
            raise ValueError('commit readback mismatch; no local reclamation')
        if validation:
            marker = {'edition': EDITION, 'worker': self.args.worker,
                      'assignment_sha256': self.assignment_sha, 'code_signatures': self.signatures,
                      'indices': [0, 4, 8], 'commit_path': str(target / 'commit.json'),
                      'commit_sha256': digest(commit_raw), 'owner_pid': os.getpid(),
                      'verified_unix': time.time(), 'gpu_uuid': self.gate.uuid}
            self.verify_first3(marker)
            self.core.write_once(self.marker, raw_json(marker))
            self.access.publish_json(marker, str(self.shared / 'first3_verified.json'))
        self.progress['published'] += commit['unique_sources_attempted']
        self.progress['published_generated_with_events'] += commit['unique_sources_with_action_events']
        self.progress['profile_seconds']['pack_publish'] = self.progress['profile_seconds'].get('pack_publish', 0) + time.perf_counter() - started
        self.core.write_once(local_batch / 'published_verified.json', raw_json({'commit_path': str(target / 'commit.json'), 'commit_sha256': digest(commit_raw), 'run_id': self.run_id}))
        for path in self.inputs:
            if path.is_relative_to(self.work / 'staging'):
                shutil.rmtree(path)
        shutil.rmtree(local_batch)
        self.inputs, self.receipts, self.output_used = [], [], 0
        self.batch_no += 1
        self.report(stage='batch_published', first3_verified=validation or self.marker.exists())

    def run(self):
        if self.args.gpu:
            if not self.marker.is_file():
                raise PermissionError('own node current-edition first3 marker absent; no model/claim/lock')
            self.verify_first3(json.loads(self.marker.read_bytes()))
        elif self.args.resume_first3:
            if self.args.worker != 'a100-extra-03' or not self.marker.is_file():
                raise PermissionError('continuation requires03 own published first3 evidence')
            self.verify_first3(json.loads(self.marker.read_bytes()))
            self.next = 3
            self.progress['prior_published_sources_excluded'] = 3
            self.progress['excluded_assignment_indices'] = [0, 4, 8]
        elif self.marker.exists():
            raise PermissionError('GPU0 first3 already exists; no automatic reuse/retry')
        self.work.mkdir(parents=True, exist_ok=False)
        with Gate(self.args, self.assignment_sha, self.helper, self.work) as self.gate:
            sys.path.insert(0, str(self.args.code_root))
            self.core = load_module('unified_algorithm_core', self.args.code_root / 'ms_unified_novlm_v1.py')
            self.core.write_once(self.work / 'resource_evidence.json', raw_json(self.stage()))
            self.report(status='models_loading', code_signatures=self.signatures)
            monitor = threading.Thread(target=self.heartbeat, daemon=True); monitor.start()
            try:
                started = time.perf_counter()
                self.models = self.core.load_models({'gpu_index': self.args.gpu,
                    'scail_root': str(self.args.scail_root), 'beat_checkpoint': str(self.args.beat_checkpoint)})
                self.progress['profile_seconds']['models_load_once'] = time.perf_counter() - started
                with ThreadPoolExecutor(max_workers=1) as executor:
                    boundary = 3 if self.args.gpu == 0 and not self.args.resume_first3 else 250
                    while self.next < 250 or self.pending:
                        self.stage()
                        while self.enqueue(boundary, executor):
                            pass
                        if self.pending:
                            self.process()
                        if self.receipts and (len(self.receipts) >= (3 if boundary == 3 else 20)
                                or (not self.pending and self.next >= boundary)
                                or not self.pending):
                            if boundary == 3 and len(self.receipts) != 3:
                                self.publish(False)
                                raise ValueError('first3 bounded cache ended before all three; no expansion')
                            self.publish(boundary == 3)
                            if boundary == 3:
                                boundary = 250
                    if self.receipts:
                        self.publish(False)
                self.report(status='completed', stage='final', ended_unix=time.time())
            except BaseException as error:
                self.report(status='failed_sealed_no_retry', error=str(error), ended_unix=time.time())
                raise
            finally:
                self.stop.set(); monitor.join()

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--worker', required=True, choices=[f'a100-extra-{i:02d}' for i in range(3, 7)])
    parser.add_argument('--gpu', required=True, type=int, choices=range(4))
    parser.add_argument('--ticket', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--code-root', required=True, type=Path)
    parser.add_argument('--scail-root', required=True, type=Path)
    parser.add_argument('--beat-checkpoint', required=True, type=Path)
    parser.add_argument('--relay-host')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--resume-first3', action='store_true', help='explicit03 continuation excludes already published0/4/8;preserves sealed previous lease')
    parser.add_argument('--recover-preflight', action='store_true', help='03 prior resource gate refused with0sourceclaims; preserve old lease and check fresh gate again')
    args = parser.parse_args(argv)
    if args.recover_preflight and args.worker != 'a100-extra-03':
        parser.error('preflight continuation limited to03 verified0claim records')
    if args.resume_first3 and (args.worker != 'a100-extra-03' or args.gpu != 0):
        parser.error('resume-first3 only03/GPU0; no consumed source retry')
    for key in ('root', 'code_root', 'scail_root', 'beat_checkpoint'):
        if not getattr(args, key).is_absolute():
            parser.error('absolute local paths required')
    if args.code_root != args.root / 'unified_code_snapshot_v1' or args.scail_root != args.root / 'unified_scail_local_v1' or args.beat_checkpoint != args.root / 'model_cache/BeatThis/final0.verified.ckpt':
        parser.error('new-edition local snapshot/SCAIL/verified BeatThis paths required')
    from unified_access_transport_v3 import Access
    access = Access(args.worker, relay_host=args.relay_host)
    ticket, rows, sha, signatures, helper = verify_assignment(args, access)
    if args.check_only:
        print(json.dumps({'status': 'check_only_verified_no_models_claims_locks', 'worker': args.worker,
                          'gpu': args.gpu, 'assignment_sha256': sha, 'lane_count': 250,
                          'code_signatures': signatures}), flush=True)
        return 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    Lane(args, access, ticket, rows, sha, signatures, helper).run()
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
