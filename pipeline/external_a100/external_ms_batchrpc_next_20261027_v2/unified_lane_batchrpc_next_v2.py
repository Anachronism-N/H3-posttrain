"""Safe terminal-boundary queue rollover; unchanged core; fresh PID loads models once."""
from __future__ import annotations

import argparse, fcntl, hashlib, importlib.util, io, json, os, re, resource
import shutil, shlex, subprocess, sys, tarfile, tempfile, threading, time, uuid
import xml.etree.ElementTree as ET
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GIB = 2 ** 30
BUDGET, OUTPUT_CAP, INPUT_CAP = 8 * GIB, 4 * GIB, 2 * GIB
OLD_COMMON = Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/external_ms_unified_run_20261027_v1')
COMMON = OLD_COMMON.parent / 'external_ms_unified_clusters80k_20261027_v1'
EDITION = COMMON.name
CORE_SHA = '85ea6781a433a7f3cfe50fa074f56e7d7c291a815dfbf7b0220d4eb9caf8dda9'
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

def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key; cursor retained at zero')
            result[key] = value
        return result
    def bad(value):
        raise ValueError('nonfinite JSON; cursor retained at zero')
    if not raw or b'\0' in raw:
        raise ValueError('empty/NUL JSON; cursor retained at zero')
    return json.loads(raw.decode('utf-8', 'strict'), object_pairs_hook=pairs, parse_constant=bad)

def pid_alive(record):
    pid, ticks = record.get('pid'), record.get('start_ticks')
    if type(pid) is not int or pid <= 0 or type(ticks) is not int or ticks <= 0:
        raise PermissionError('missing actual old PID/start identity')
    try:
        return start_ticks(pid) == ticks
    except FileNotFoundError:
        return False

def previous_local(args):
    expected = args.root / 'logs' / f'unified_gpu{args.gpu}_progress.json'
    if args.previous_progress != expected:
        raise PermissionError('original own old progress path required')
    raw = expected.read_bytes()
    previous = strict_json(raw)
    if (previous.get('worker') != args.worker or previous.get('physical_gpu_index') != args.gpu
            or previous.get('status') not in ('completed', 'failed_sealed_no_retry')
            or not previous.get('ended_unix') or pid_alive(previous)):
        raise PermissionError('old lane not terminal with ended PID/start; no models/claims/locks')
    paths = sorted((args.root / 'gpu_locks').glob(f'physical_gpu{args.gpu}*.lease.json'))
    original = args.root / 'gpu_locks' / f'physical_gpu{args.gpu}.lease.json'
    if original not in paths:
        raise PermissionError('original physical lease absent; never fabricate rollover authorization')
    evidence, matched = {}, False
    for path in paths:
        lease_raw = path.read_bytes()
        lease = strict_json(lease_raw)
        if pid_alive(lease):
            raise PermissionError('old or foreign lease PID/start is active; unchanged')
        if path.name.endswith('.newepoch.lease.json'):
            raise PermissionError('prior rollover epoch is sealed; no automatic retry')
        if lease.get('worker') != args.worker or lease.get('physical_gpu_index') != args.gpu:
            raise PermissionError('foreign physical lease; unchanged')
        evidence[str(path)] = digest(lease_raw)
        if lease.get('pid') == previous['pid'] and lease.get('start_ticks') == previous['start_ticks']:
            if (lease.get('status') != 'released_sealed_no_retry'
                    or lease.get('assignment_sha256') != previous.get('assignment_sha256')):
                raise PermissionError('old final lease not released consistently with progress')
            matched = True
    if not matched:
        raise PermissionError('old terminal PID/start final lease absent')
    if (args.root / 'logs' / f'rollover_gpu{args.gpu}_progress.json').exists():
        raise PermissionError('existing new progress retained; no automatic retry')
    return previous, {'previous_progress_path': str(expected), 'previous_progress_sha256': digest(raw),
                      'preserved_old_lease_sha256': evidence}

def verify_queue(args, access):
    if Path(args.queue) != COMMON / (args.worker + '.queue.json'):
        raise PermissionError('original assigned queue path required')
    raw = access.read(args.queue)
    queue = strict_json(raw)
    flags = ('automatic_rollover_within_preallocated_tickets', 'no_automatic_retry_sealed_previous_sources',
             'no_restart_active_previous_job', 'run_until_preallocated_queue_exhausted')
    if (queue.get('schema') != 'ms_external_authorized_rollover_queue_v1'
            or queue.get('worker_id') != args.worker or any(queue.get(k) is not True for k in flags)
            or queue.get('request_more_when_remaining_below') != 1000):
        raise PermissionError('original finite rollover queue policy mismatch')
    entries = queue.get('tickets')
    old_path = str(OLD_COMMON / 'assignments' / (args.worker + '.ticket.json'))
    if not isinstance(entries, list) or len(entries) != 2 or [e.get('path') for e in entries] != [old_path, args.ticket]:
        raise PermissionError('only preallocated old-consumed/new20k tickets authorized')
    if any(set(e) != {'path', 'sha256'} or not re.fullmatch('[0-9a-f]{64}', str(e['sha256'])) for e in entries):
        raise PermissionError('queue ticket SHA contract')
    args.queue_sha = digest(raw)
    return queue

def old_commit_index(args):
    script = """
import hashlib, json, pathlib, re, sys
worker, gpu = sys.argv[1:]
if worker not in ('a100-extra-03', 'a100-extra-04', 'a100-extra-05', 'a100-extra-06') or gpu not in ('0','1','2','3'):
    raise PermissionError('old commit index own worker/GPU only')
base = pathlib.Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/external_ms_unified_run_20261027_v1') / 'results' / worker / 'batches' / ('gpu'+gpu)
if base.resolve(strict=True) != base:
    raise PermissionError('redirected old own publication directory')
paths = sorted(base.glob('*/commit.json'))
if len(paths) > 500:
    raise PermissionError('old lane metadata index exceeds finite bound')
result = []
for path in paths:
    if not re.fullmatch('[A-Za-z0-9_-]+', path.parent.name) or path.resolve(strict=True) != path:
        raise PermissionError('unsafe redirected old commit')
    before = path.stat()
    if not path.is_file() or before.st_size > 4*1024*1024:
        raise PermissionError('old commit metadata size/type')
    raw = path.read_bytes()
    after = path.stat()
    fields = ('st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_ino', 'st_dev')
    if any(getattr(before, k) != getattr(after, k) for k in fields) or len(raw) != before.st_size or b'\\0' in raw:
        raise PermissionError('old commit metadata changed/NUL')
    result.append({'path':str(path), 'sha256':hashlib.sha256(raw).hexdigest()})
print(json.dumps(result))
"""
    command = [sys.executable, '-c', script, args.worker, str(args.gpu)]
    if args.relay_host:
        from unified_access_transport_v3 import RELAY_HOST
        if args.relay_host != RELAY_HOST:
            raise PermissionError('old index relay not allowlisted')
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                   '-o', 'UserKnownHostsFile=/tmp/ms_pilot_relay_known_hosts', '-o', 'ConnectTimeout=15',
                   args.relay_host, shlex.join(['python3', '-c', script, args.worker, str(args.gpu)])]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=120)
    if len(result.stdout) > 4 * 1024 ** 2:
        raise PermissionError('old metadata index oversized')
    return strict_json(result.stdout)

def verify_models(args):
    frozen = {'DWPose/yolox_l.onnx': '7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411',
              'DWPose/dw-ll_ucoco_384.onnx': '724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843',
              'BeatThis/final0.verified.ckpt': '8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331'}
    identity_raw = (args.root / 'model_cache/cache_identity.json').read_bytes()
    identity = strict_json(identity_raw)
    if identity.get('schema') != 'local_model_cache_identity_v1' or identity.get('local_paths_revalidated') is not True:
        raise PermissionError('old revalidated local model cache identity absent')
    entries = {r['relative_path']: r for r in identity['files']}
    for relative, sha in frozen.items():
        path = args.root / 'model_cache' / relative
        entry = entries.get(relative, {})
        if (entry.get('sha256') != sha or entry.get('verified') is not True or entry.get('local_path') != str(path)
                or path.stat().st_size != entry.get('bytes') or file_sha(path) != sha):
            raise PermissionError('actual models changed from old verified frozen cache')
        if relative.startswith('DWPose/'):
            bound = args.scail_root / 'pretrained_weights' / relative
            if bound.resolve(strict=True) != path.resolve(strict=True):
                raise PermissionError('core model path does not resolve to old verified local model')
    provenance = strict_json((args.root / 'model_cache/BeatThis/local_source_manifest.json').read_bytes())
    if (provenance.get('checkpoint_path') != str(args.beat_checkpoint)
            or provenance.get('checkpoint_sha256') != frozen['BeatThis/final0.verified.ckpt']
            or provenance.get('model_bytes_unchanged') is not True):
        raise PermissionError('actual local BeatThis final0 source binding changed')
    return {'model_signatures': frozen, 'old_cache_identity_sha256': digest(identity_raw),
            'old_verified_models_unchanged': True}

def verify_published_fields(commit, commit_path):
    target = Path(commit_path).parent
    files = {}
    for item in commit.get('publications', []):
        path, sha, receipt = item.get('path'), item.get('sha256'), item.get('receipt', {})
        if (not isinstance(path, str) or '..' in Path(path).parts or str(Path(path)) != path
                or not Path(path).is_relative_to(target) or path in files
                or not re.fullmatch('[0-9a-f]{64}', str(sha)) or receipt.get('path') != path
                or receipt.get('sha256') != sha or type(receipt.get('bytes')) is not int or receipt['bytes'] <= 0):
            raise PermissionError('old published path/SHA/byte receipt fields mismatch')
        files[path] = sha
    if files.get(str(target / 'batch.tar')) != commit.get('batch_sha256'):
        raise PermissionError('old committed archive publication SHA mismatch')
    for receipt in commit.get('receipts', []):
        if receipt.get('status') == 'existing_claim_not_reassigned':
            continue
        row = receipt['source_row']
        case = target / 'cases' / row['split'] / row['id']
        names = ('receipt.json',) if receipt.get('status') == 'failed_sealed_no_retry' else (
            'pose.npz', 'fullframe_manifest.json', 'beats.json', 'draft.json', 'receipt.json')
        if any(str(case / name) not in files for name in names):
            raise PermissionError('old sealed source required publication fields absent')

def verify_previous(args, access, queue, previous, evidence, ticket, rows, signatures):
    from unified_access_transport_v3 import Access as OldAccess
    old = OldAccess(args.worker, relay_host=args.relay_host)
    old_path = queue['tickets'][0]['path']
    old_raw = old.read(old_path)
    old_ticket = strict_json(old_raw)
    old_manifest_raw = old.read(old_ticket['assignment_path'])
    if digest(old_raw) != queue['tickets'][0]['sha256'] or digest(old_manifest_raw) != old_ticket['assignment_sha256']:
        raise PermissionError('consumed old original ticket/manifest SHA mismatch')
    if not old_manifest_raw.endswith(b'\n') or b'\0' in old_manifest_raw:
        raise PermissionError('old original manifest incomplete/NUL')
    old_rows = [strict_json(line) for line in old_manifest_raw.split(b'\n')[:-1]]
    old.bind(old_ticket, old_rows, digest(old_raw), digest(old_manifest_raw))
    if (previous.get('assignment_sha256') != old_ticket['assignment_sha256'] or ticket['code_refs'] != old_ticket['code_refs']
            or ticket.get('model_policy') != old_ticket.get('model_policy')):
        raise PermissionError('new code references differ from consumed old verified assignment')
    old_ids = {(r['split'].casefold(), r['id'].casefold()) for r in old_rows}
    old_media = {r[k] for r in old_rows for k in ('video_path', 'audio_path')}
    if any((r['split'].casefold(), r['id'].casefold()) in old_ids
           or any(r[k] in old_media for k in ('video_path', 'audio_path')) for r in rows):
        raise PermissionError('new manifest contains consumed old sources; never retry old failure budget')
    marker_path = args.root / 'unified_runs' / OLD_COMMON.name / 'first3_verified.json'
    marker_raw = marker_path.read_bytes()
    marker = strict_json(marker_raw)
    marker_expected = {'edition': OLD_COMMON.name, 'worker': args.worker,
                       'assignment_sha256': old_ticket['assignment_sha256'], 'code_signatures': signatures,
                       'indices': [0, 4, 8]}
    if any(marker.get(k) != v for k, v in marker_expected.items()):
        raise PermissionError('real old own first3 marker/code signatures mismatch')
    if old.read(str(Path(old_ticket['output_root']) / 'first3_verified.json'), expected_sha=digest(marker_raw)) != marker_raw:
        raise PermissionError('old published first3 marker differs from actual local bytes')
    commit_raw = old.read(marker['commit_path'], expected_sha=marker['commit_sha256'])
    commit = strict_json(commit_raw)
    if (not Path(marker['commit_path']).is_relative_to(Path(old_ticket['output_root']) / 'batches/gpu0')
            or commit.get('edition') != OLD_COMMON.name or commit.get('worker') != args.worker
            or commit.get('assignment_sha256') != old_ticket['assignment_sha256']
            or commit.get('code_signatures') != signatures or commit.get('validation_first3') is not True
            or commit.get('assignment_indices') != [0, 4, 8] or commit.get('unique_sources_attempted') != 3
            or len(commit.get('receipts', [])) != 3 or not commit.get('publications')
            or any(not r.get('generated_with_action_events') for r in commit['receipts'])):
        raise PermissionError('old published first3 small commit proof invalid; no repeated payload/GPU validation')
    verify_published_fields(commit, marker['commit_path'])
    own_published, observed = 0, set()
    commits = old_commit_index(args)
    for item in commits:
        c = strict_json(old.read(item['path'], expected_sha=item['sha256']))
        if (c.get('edition') != OLD_COMMON.name or c.get('worker') != args.worker or c.get('gpu') != args.gpu
                or c.get('assignment_sha256') != old_ticket['assignment_sha256'] or c.get('code_signatures') != signatures):
            raise PermissionError('old lane commit identity mismatch')
        receipts = c.get('receipts', [])
        if c.get('assignment_indices') != [r.get('assignment_index') for r in receipts]:
            raise PermissionError('old publication assignment indices mismatch')
        attempted = 0
        for r in receipts:
            index = r.get('assignment_index')
            if type(index) is not int or not 0 <= index < len(old_rows) or index % 4 != args.gpu or index in observed:
                raise PermissionError('old commit wrong lane or duplicate original source')
            observed.add(index)
            if r.get('source_row') != old_rows[index]:
                raise PermissionError('old receipt differs from immutable source row')
            attempted += r.get('status') != 'existing_claim_not_reassigned'
        if attempted != c.get('unique_sources_attempted') or not c.get('publications'):
            raise PermissionError('old publication count/fields mismatch')
        verify_published_fields(c, item['path'])
        if c.get('run_id') == previous['run_id']:
            own_published += attempted
    if own_published != previous.get('published') or (previous['status'] == 'completed'
            and len(observed) != len(old_rows) // 4):
        raise PermissionError('old final publication commits not consistent with terminal progress')
    evidence.update(ticket_path=old_path, ticket_sha256=digest(old_raw), assignment_sha256=old_ticket['assignment_sha256'],
                    status=previous['status'], pid=previous['pid'], start_ticks=previous['start_ticks'],
                    source_count=len(old_rows), lane_sources_excluded=len(old_rows) // 4,
                    prior_published=own_published, old_commit_count=len(commits),
                    old_first3_marker_sha256=digest(marker_raw), old_first3_commit_sha256=digest(commit_raw),
                    no_automatic_retry=True)
    return evidence

def verify_assignment(args, access, queue):
    ticket_raw, assignment_raw = access.read(args.ticket), access.read(args.manifest)
    ticket, assignment_sha = strict_json(ticket_raw), digest(assignment_raw)
    if digest(ticket_raw) != next(item['sha256'] for item in queue['tickets'] if item['path'] == args.ticket):
        raise ValueError('new ticket differs from authorized queue SHA')
    if (ticket.get('schema') != 'ms_external_unified_assignment_v1'
            or ticket.get('worker_id') != args.worker or ticket.get('edition') != EDITION
            or type(ticket.get('source_count')) is not int or ticket.get('source_count') != 20000 or ticket.get('assignment_sha256') != assignment_sha
            or ticket.get('allowed_gpu_indices') != [0, 1, 2, 3]
            or Path(args.ticket) != COMMON / 'assignments' / (args.worker + '.ticket.json')
            or Path(args.manifest) != COMMON / 'assignments' / (args.worker + '.jsonl')
            or ticket.get('assignment_path') != args.manifest
            or Path(ticket.get('output_root', '')) != COMMON / 'results' / args.worker
            or Path(ticket.get('claims_root', '')) != COMMON / 'claims'):
        raise ValueError('original own-worker ticket/assignment contract mismatch')
    if not assignment_raw or bytes([0]) in assignment_raw or not assignment_raw.endswith(bytes([10])):
        raise ValueError('NUL or partial assignment; cursor retained at zero')
    rows = [strict_json(line) for line in assignment_raw.split(bytes([10]))[:-1]]
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
    if len(rows) != ticket['source_count'] or len(rows) % 4:
        raise ValueError('exactly 20000 original source rows required')
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
    expected_startup = ('Continue same validated BGR/fullframe/CPU-rules pipeline without repeat3case tests for unchanged code/models. '
                        'New GPU process requires fresh local resource admission. Do not hot-update executing version.')
    if (ticket.get('startup_policy') != expected_startup or ticket.get('automatic_retry') is not False
            or ticket.get('models_reused') is not True or ticket.get('weight_load_once_per_GPU_process') is not True
            or ticket.get('resource_scope_unchanged') is not True
            or ticket.get('source_intersection_with_all_prior_allocations') != 0
            or ticket.get('publication_sources_per_batch') != 20):
        raise PermissionError('unchanged-code/models no-repeat3 and finite no-retry policy required')
    access.bind(ticket, rows, digest(ticket_raw), assignment_sha)
    return ticket, rows, assignment_sha, signatures, load_module('unified_memory_helper', helper)

class Gate:
    def __init__(self, args, assignment_sha, helper, run_root):
        self.args, self.helper, self.run_root = args, helper, run_root
        self.lock_dir = args.root / 'gpu_locks'
        self.lease = self.lock_dir / f'physical_gpu{args.gpu}.newepoch.lease.json'
        self.identity = {'pid': os.getpid(), 'start_ticks': start_ticks(os.getpid()),
                         'worker': args.worker, 'physical_gpu_index': args.gpu,
                         'assignment_sha256': assignment_sha, 'edition': EDITION,
                         'queue_sha256': args.queue_sha, 'lease_id': uuid.uuid4().hex}
        self.fd, self.uuid, self.peak, self.created = None, None, 0, False
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
        if fresh and used > 4:
            raise PermissionError('fresh_memory_above_transient_residue; unchanged')
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
            for path, sha in self.args.previous_evidence['preserved_old_lease_sha256'].items():
                if file_sha(path) != sha or pid_alive(strict_json(Path(path).read_bytes())):
                    raise PermissionError('old lease bytes changed or old PID/start active; unchanged')
            if file_sha(self.args.previous_progress) != self.args.previous_evidence['previous_progress_sha256']:
                raise PermissionError('old terminal progress changed; untouched')
            current = strict_json(self.lease.read_bytes())
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
            for path in self.lock_dir.glob(f'physical_gpu{self.args.gpu}*.lease.json'):
                if pid_alive(strict_json(path.read_bytes())):
                    raise PermissionError('active foreign physical lease PID/start; unchanged')
            if self.lease.exists():
                raise PermissionError('existing lease retained; no overwrite or automatic retry')
            fd = os.open(self.lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            self.created = True
            with os.fdopen(fd, 'wb') as f:
                f.write(raw_json(self.identity)); f.flush(); os.fsync(f.fileno())
            evidence = self.check(True)
            self.uuid = evidence['gpu_uuid']
            return self
        except BaseException as error:
            if self.created:
                atomic_json(self.lease, {**self.identity, 'gpu_uuid': self.uuid,
                                        'status': 'released_sealed_no_retry', 'error': str(error),
                                        'ended_unix': time.time()})
            os.close(self.fd); self.fd = None
            raise

    def __exit__(self, *exc):
        if self.fd is not None:
            with self.mutex:
                atomic_json(self.lease, {**self.identity, 'gpu_uuid': self.uuid,
                                        'status': 'released_sealed_no_retry', 'ended_unix': time.time()})
                os.close(self.fd); self.fd = None

def load_frozen_core(args):
    path = args.code_root / 'ms_unified_novlm_v1.py'
    if file_sha(path) != CORE_SHA: raise ValueError('frozen original core SHA mismatch')
    sys.path.insert(0, str(args.code_root))
    return load_module('unified_algorithm_core', path)

def recover_publication_only(access, local_batch):
    # This entry has no model, source claim, GPU gate/query or generation callbacks.
    saved = strict_json((Path(local_batch) / 'publication_plan.json').read_bytes())
    return saved, access.recover_prepared(saved)

class Lane:
    def __init__(self, args, access, ticket, rows, assignment_sha, signatures, helper):
        self.args, self.access, self.ticket = args, access, ticket
        self.assignment_sha, self.signatures, self.helper = assignment_sha, signatures, helper
        self.original_rows = rows
        self.rows = [(i, r) for i, r in enumerate(rows) if i % 4 == args.gpu]
        assert len(self.rows) == ticket['source_count'] // 4
        self.run_id = f'g{args.gpu}-{time.time_ns()}-{uuid.uuid4().hex[:8]}'
        self.edition_root = args.root / 'unified_runs' / EDITION
        self.work = self.edition_root / self.run_id
        self.marker = args.root / 'unified_runs' / OLD_COMMON.name / 'first3_verified.json'
        self.shared = Path(ticket['output_root'])
        self.pending, self.receipts, self.inputs = deque(), [], []
        self.next, self.batch_no, self.reserved, self.output_used = 0, 0, 0, 0
        self.stop, self.fatal = threading.Event(), None
        self.progress_path = args.root / 'logs' / f'rollover_gpu{args.gpu}_progress.json'
        self.progress = {'worker': args.worker, 'physical_gpu_index': args.gpu,
                         'assignment_sha256': assignment_sha, 'run_id': self.run_id,
                         'pid': os.getpid(), 'start_ticks': start_ticks(os.getpid()),
                         'started_unix': time.time(), 'unique_processed': 0, 'published': 0,
                         'queue_path': args.queue, 'queue_sha256': args.queue_sha,
                         'old_consumed_ticket': args.previous_evidence,
                         'new_process_initialization': True, 'models_reused_across_old_pid': False,
                         'lane_source_count': len(self.rows), 'repeat_first3': False,
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

    def enqueue(self, boundary, executor):
        if self.next >= boundary or len(self.pending) >= 2 or len(self.receipts) + len(self.pending) >= 20:
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
        hashes, audit = {}, None
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
            audit = self.models[1]
            audit.configure(row, index, hashes, case)
            result = self.core.extract_and_rule(row, copied, hashes, self.models, case,
                       target / 'cases' / row['split'] / row['id'], probe=probe,
                       max_output_bytes=remaining, audio_info=audit.audio_info, profile=profile, source_started=started)
            draft = json.loads((case / 'draft.json').read_bytes())
            result['event_count'] = len(draft['events'])
            result['assignment_index'] = index
            result['copy_evidence'] = self.access.copy_evidence.get((row['split'], row['id'])) or {
                'mode': 'direct_core_copy_source_one_pass', 'input_sha256': hashes,
                'original_paths': {k: row[k] for k in copied}, 'copied_bytes': size}
            result['generated_with_action_events'] = bool(draft['events'])
            atomic_json(case / 'receipt.json', result)
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
                      'error': str(error), 'profile_seconds': profile, **self.core.v2.FLAGS, 'vlm_called': False}
            if hashes.get('audio_path'): result['actual_audio_sha256'] = hashes['audio_path']
            if audit is not None and audit.metadata is not None: result['actual_audio_metadata'] = audit.metadata
            if audit is not None and audit.result.get('diagnostic_paths'):
                result['failed_predicates'] = audit.result['failed_predicates']
                result['diagnostic_publications'] = [dict(path=str(target / Path(path).relative_to(local_batch)), bytes=Path(path).stat().st_size, sha256=file_sha(path)) for path in audit.result['diagnostic_paths']]
            if audit is not None and audit.gap: result['evidence_gap'] = audit.gap
            self.core.write_once(case / 'receipt.json', raw_json(result))
            self.progress['failed'] += 1
        self.output_used = tree_bytes(local_batch)
        self.receipts.append(result); self.progress['unique_processed'] += 1
        for key, value in profile.items():
            totals = self.progress['profile_seconds']; totals[key] = totals.get(key, 0) + value
        self.report(status='running', stage='source_complete')
        if self.output_used > OUTPUT_CAP or tree_bytes(self.work) + self.reserved + self.output_used > BUDGET:
            raise ValueError('cache boundary exceeded; retained sealed outputs, no expansion')

    def publish(self, validation):
        from batchrpc_publication_next_v2 import prepare, save_once
        local_batch, target = self.batch_paths()
        local_batch.mkdir(parents=True, exist_ok=True)
        archive = local_batch / 'batch.tar'
        plan_path = local_batch / 'publication_plan.json'
        started = time.perf_counter()
        if plan_path.exists():
            saved = strict_json(plan_path.read_bytes())
            expected_indices = [r.get('assignment_index') for r in self.receipts]
            if (saved['plan']['target'] != str(target) or saved['plan']['commit'].get('run_id') != self.run_id
                    or saved['plan']['commit'].get('assignment_indices') != expected_indices):
                raise ValueError('local immutable prepared plan identity mismatch')
            result = self.access.recover_prepared(saved)
        else:
            self.stage()
            if tree_bytes(self.work) + self.reserved + self.output_used + 1024 ** 2 > BUDGET:
                raise ValueError('archive duplication exceeds 8GiB; preserve generated artifacts')
            with tarfile.open(archive, 'x') as pack:
                for case in sorted((local_batch / 'cases').glob('*/*')):
                    for name in self.core.FILES:
                        path = case / name
                        if path.is_file(): pack.add(path, arcname=str(path.relative_to(local_batch)), recursive=False)
            with archive.open('rb') as stream: os.fsync(stream.fileno())
            commit = {'schema': 'ms_unified_novlm_batch_commit_v1', 'edition': EDITION,
                      'worker': self.args.worker, 'gpu': self.args.gpu, 'run_id': self.run_id,
                      'assignment_sha256': self.assignment_sha, 'code_signatures': self.signatures,
                      'next_version_code_signatures': {name: file_sha(Path(__file__).parent / name) for name in (
                          'unified_lane_batchrpc_next_v2.py', 'unified_access_transport_batchrpc_next_v2.py',
                          'batchrpc_publication_next_v2.py', 'beatthis_rejection_next_v2.py')},
                      'assignment_indices': [r.get('assignment_index') for r in self.receipts],
                      'receipts': self.receipts, 'validation_first3': validation, 'quality_approved': False,
                      'unique_sources_attempted': sum(r['status'] != 'existing_claim_not_reassigned' for r in self.receipts),
                      'unique_sources_with_action_events': sum(bool(r.get('generated_with_action_events')) for r in self.receipts),
                      'frame_count': sum(r.get('frame_count', 0) for r in self.receipts),
                      'profile_seconds': {'pack_publish': time.perf_counter() - started}, **self.core.v2.FLAGS,
                      'vlm_called': False, 'implemented_not_deployed': True}
            saved = prepare(local_batch, target, commit)
            self.stage()
            result = self.access.publish_prepared(saved)
        commit = saved['plan']['commit']
        if result['verification']['state'] != 'committed_verified':
            raise ValueError('unverified commit; no publication count or local reclamation')
        commit_raw = self.access.read(str(target / 'commit.json'), expected_sha=saved['plan']['commit_sha256'])
        if strict_json(commit_raw) != commit: raise ValueError('commit readback mismatch; retain cache')
        marker = local_batch / 'published_verified.json'
        if not marker.exists():
            save_once(marker, raw_json({'commit_path': str(target / 'commit.json'),
                'commit_sha256': digest(commit_raw), 'run_id': self.run_id, 'publication_state': result['state']}))
        accounted = self.progress.setdefault('verified_batch_commits', {})
        key = str(target / 'commit.json')
        if key in accounted and accounted[key] != saved['plan']['commit_sha256']:
            raise ValueError('accounted commit SHA conflict; no duplicate count or reclamation')
        if key not in accounted:
            self.progress['published'] += commit['unique_sources_attempted']
            self.progress['published_generated_with_events'] += commit['unique_sources_with_action_events']
            accounted[key] = saved['plan']['commit_sha256']
        self.progress['profile_seconds']['pack_publish'] = self.progress['profile_seconds'].get('pack_publish', 0) + time.perf_counter() - started
        self.report(stage='batch_commit_verified_before_reclamation')
        for path in self.inputs:
            if path.is_relative_to(self.work / 'staging'): shutil.rmtree(path)
        shutil.rmtree(local_batch)
        self.inputs, self.receipts, self.output_used = [], [], 0
        self.batch_no += 1
        self.report(stage='batch_published', previous_first3_verified=True)

    def refresh_queue(self):
        queue = verify_queue(self.args, self.access)
        if queue['tickets'] != self.args.original_queue['tickets']:
            raise PermissionError('queue replaced consumed/preallocated references; no unapproved ticket')
        self.progress['queue_sha256'] = self.args.queue_sha
        if len(self.rows) - self.next < queue['request_more_when_remaining_below']:
            self.progress['requests_more_authorized_sources'] = True

    def run(self):
        self.work.mkdir(parents=True, exist_ok=False)
        self.gate = Gate(self.args, self.assignment_sha, self.helper, self.work)
        with self.gate:
            self.core = load_frozen_core(self.args)
            self.core.write_once(self.work / 'resource_evidence.json', raw_json(self.stage()))
            self.report(status='models_loading', code_signatures=self.signatures)
            monitor = threading.Thread(target=self.heartbeat, daemon=True); monitor.start()
            try:
                started = time.perf_counter()
                self.models = self.core.load_models({'gpu_index': self.args.gpu,
                    'scail_root': str(self.args.scail_root), 'beat_checkpoint': str(self.args.beat_checkpoint)})
                from beatthis_rejection_next_v2 import AuditTap
                self.models = (self.models[0], AuditTap(self.models[1], self.core.audio_metadata))
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
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--worker', required=True, choices=[f'a100-extra-{i:02d}' for i in range(3, 7)])
    parser.add_argument('--gpu', required=True, type=int, choices=range(4))
    parser.add_argument('--ticket', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--previous-progress', required=True, type=Path)
    parser.add_argument('--code-root', required=True, type=Path)
    parser.add_argument('--scail-root', required=True, type=Path)
    parser.add_argument('--beat-checkpoint', required=True, type=Path)
    parser.add_argument('--relay-host')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    parser.error('implemented_not_deployed: no signed next-version epoch handoff or fresh GPU admission; production execution disabled')
    for key in ('root', 'code_root', 'scail_root', 'beat_checkpoint', 'previous_progress'):
        if not getattr(args, key).is_absolute():
            parser.error('absolute local paths required')
    if (args.code_root != args.root / 'unified_code_snapshot_v1'
            or args.scail_root != args.root / 'unified_scail_local_v1'
            or args.beat_checkpoint != args.root / 'model_cache/BeatThis/final0.verified.ckpt'):
        parser.error('unchanged old verified local snapshot/SCAIL/BeatThis paths required')
    previous, evidence = previous_local(args)
    from unified_access_transport_batchrpc_next_v2 import Access
    access = Access(args.worker, relay_host=args.relay_host)
    queue = verify_queue(args, access)
    ticket, rows, sha, signatures, helper = verify_assignment(args, access, queue)
    evidence.update(verify_models(args))
    args.previous_evidence = verify_previous(args, access, queue, previous, evidence, ticket, rows, signatures)
    args.original_queue = queue
    if args.check_only:
        print(json.dumps({'status': 'check_only_verified_no_models_claims_locks', 'worker': args.worker,
                          'gpu': args.gpu, 'assignment_sha256': sha, 'lane_count': len(rows) // 4,
                          'queue_sha256': args.queue_sha, 'repeat_first3': False,
                          'new_process_initialization': True, 'code_signatures': signatures,
                          'old_consumed_ticket': args.previous_evidence}), flush=True)
        return 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    lane = Lane(args, access, ticket, rows, sha, signatures, helper)
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
