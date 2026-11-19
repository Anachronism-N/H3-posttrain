"""Publication-only recovery of four sealed original batch176 caches; no inference/claims."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import stat
import sys
import tarfile
import time

sys.dont_write_bytecode = True
L = Path('/tmp/ms_pose2d_draft_20261026T1427Z/launcher')
P = Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/external_ms_unified_clusters80k_20261027_v1')
CANDIDATE = L / 'recovery03_publication_v1'
WORKER = 'a100-extra-03'
RUNS = ('g0-1793043643552659367-0bb56122', 'g1-1793043640723803227-9e2f458a',
        'g2-1793043642799656714-e18e7941', 'g3-1793043643255019255-a1ab1259')
MAX_FILE = 64 * 2**20
MAX_ARCHIVE = 8 * 2**30

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()

def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def bad(value):
        raise ValueError('nonfinite JSON')
    if not raw or b'\0' in raw or not raw.endswith(b'\n'):
        raise ValueError('empty/NUL/partial JSON')
    return json.loads(raw.decode('utf-8', 'strict'), object_pairs_hook=pairs, parse_constant=bad)

def anchor(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('noncanonical evidence path')
    node = path
    while not node.exists() and not node.is_symlink():
        node = node.parent
    if node.resolve(strict=True) != node:
        raise ValueError('redirected evidence path')
    return path

def stamp(s):
    return {k: getattr(s, k) for k in ('st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_ino', 'st_dev')}

def read_evidence(path, limit=MAX_FILE):
    path = anchor(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ValueError('evidence type/size')
    with path.open('rb') as f:
        opened = f.stat() if hasattr(f, 'stat') else os.fstat(f.fileno())
        raw = f.read(limit + 1)
        after = os.fstat(f.fileno())
    if stamp(before) != stamp(opened) or stamp(opened) != stamp(after) or stamp(after) != stamp(path.stat()) or len(raw) != before.st_size:
        raise ValueError('evidence changed while reading')
    return raw, {'sha256': sha(raw), 'bytes': len(raw), **stamp(before)}

def remember(path, evidence, limit=MAX_FILE):
    raw, entry = read_evidence(path, limit)
    evidence[str(path)] = entry
    return raw

def recheck(evidence):
    for path, expected in evidence.items():
        _, actual = read_evidence(path, max(MAX_FILE, expected['bytes']))
        if actual != expected:
            raise PermissionError('preserved evidence changed: ' + path)

def write_once(path, raw):
    path = anchor(path)
    if not path.is_relative_to(CANDIDATE):
        raise PermissionError('candidate write outside independent recovery directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_evidence(path, max(MAX_ARCHIVE, len(raw)))[0] != raw:
            raise ValueError('existing candidate differs; never overwrite')
        return
    with path.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def ticks(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])

def inactive(record):
    pid, start = record.get('pid'), record.get('start_ticks')
    if type(pid) is not int or pid <= 0 or type(start) is not int or start <= 0:
        raise PermissionError('original PID/start identity missing')
    try:
        if ticks(pid) == start:
            raise PermissionError('original PID/start still active; no recovery')
    except FileNotFoundError:
        pass

def no_run_processes():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal() or int(proc.name) == os.getpid():
            continue
        try:
            cmd = (proc / 'cmdline').read_bytes()
            env = (proc / 'environ').read_bytes()
        except FileNotFoundError:
            continue
        except PermissionError as error:
            raise PermissionError('cannot exclude same-run process') from error
        if any(run.encode() in cmd or run.encode() in env for run in RUNS):
            raise PermissionError('same original run process is active: ' + proc.name)

def verify_frozen_contract(audit_raw):
    original = ast.parse(audit_raw.decode())
    recovered = ast.parse((CANDIDATE / 'frozen_deep_contract.py').read_text())
    function = next(x for x in recovered.body if isinstance(x, ast.FunctionDef) and x.name == 'validate')
    loops = [x for x in ast.walk(original) if isinstance(x, ast.For) and isinstance(x.target, ast.Name) and x.target.id == 'receipt']
    if len(loops) != 1:
        raise ValueError('original frozen audit receipt loop changed')
    loop = loops[0]
    begin = next(i for i, x in enumerate(loop.body) if isinstance(x, ast.Assign) and isinstance(x.targets[0], ast.Name) and x.targets[0].id == 'names')
    expected = loop.body[begin:]
    got_begin = next(i for i, x in enumerate(function.body) if isinstance(x, ast.Assign) and isinstance(x.targets[0], ast.Name) and x.targets[0].id == 'names')
    actual = function.body[got_begin:-1]
    if [ast.dump(x, include_attributes=False) for x in actual] != [ast.dump(x, include_attributes=False) for x in expected]:
        raise ValueError('complete original deep audit case contract not preserved')

def setup():
    evidence = {}
    for name in ('unified_lane_rollover_v1.py', 'unified_access_transport_rollover_v1.py', 'audit_rollover_output_packages_once_v2.py'):
        remember(L / name, evidence)
    verify_frozen_contract((L / 'audit_rollover_output_packages_once_v2.py').read_bytes())
    for path in (CANDIDATE / 'recover_publication.py', CANDIDATE / 'frozen_deep_contract.py'):
        remember(path, evidence)
    sys.path.insert(0, str(L))
    base = importlib.import_module('unified_lane_rollover_v1')
    access_mod = importlib.import_module('unified_access_transport_rollover_v1')
    access = access_mod.Access(WORKER)
    args = argparse.Namespace(worker=WORKER, ticket=str(P / 'assignments' / (WORKER + '.ticket.json')),
        manifest=str(P / 'assignments' / (WORKER + '.jsonl')), queue=str(P / (WORKER + '.queue.json')),
        root=L, code_root=L / 'unified_code_snapshot_v1')
    queue = base.verify_queue(args, access)
    ticket, rows, assignment_sha, signatures, _ = base.verify_assignment(args, access, queue)
    for name in signatures:
        remember(args.code_root / name, evidence)
    for path in (args.ticket, args.manifest, args.queue):
        remember(Path(path), evidence)
    sys.path.insert(0, str(args.code_root))
    rules = importlib.import_module('ms_fullframe_pose2d_draft_v2')
    # Read the core's exact constant without importing inference dependencies.
    module = ast.parse((args.code_root / 'ms_unified_novlm_v1.py').read_text())
    files = ast.literal_eval(next(x.value for x in module.body if isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FILES' for t in x.targets)))
    from frozen_deep_contract import validate
    return args, access, ticket, rows, assignment_sha, signatures, rules, tuple(files), validate, evidence

def original_gate(gpu, assignment_sha, signatures, queue_sha, evidence):
    progress_path = L / 'logs' / f'rollover_gpu{gpu}_progress.json'
    progress = strict(remember(progress_path, evidence))
    expected = {'worker': WORKER, 'physical_gpu_index': gpu, 'run_id': RUNS[gpu],
                'assignment_sha256': assignment_sha, 'code_signatures': signatures,
                'queue_path': str(P / (WORKER + '.queue.json')), 'queue_sha256': queue_sha,
                'unique_processed': 3540, 'published': 3520, 'status': 'running',
                'stage': 'source_complete', 'assignment_index': 14156 + gpu}
    if any(progress.get(k) != v for k, v in expected.items()):
        raise PermissionError('original progress identity/count/stage mismatch')
    inactive(progress)
    lease_paths = sorted((L / 'gpu_locks').glob(f'physical_gpu{gpu}*.lease.json'))
    new = L / 'gpu_locks' / f'physical_gpu{gpu}.newepoch.lease.json'
    if new not in lease_paths:
        raise PermissionError('original rollover lease missing')
    for path in lease_paths:
        lease = strict(remember(path, evidence))
        inactive(lease)
        if lease.get('worker') != WORKER or lease.get('physical_gpu_index') != gpu:
            raise PermissionError('foreign original lease')
        if path == new and any(lease.get(k) != progress[k] for k in ('worker', 'physical_gpu_index', 'assignment_sha256', 'pid', 'start_ticks', 'queue_sha256')):
            raise PermissionError('original newepoch lease/progress owner mismatch')
    old = progress['old_consumed_ticket']
    for path, expected_sha in old['preserved_old_lease_sha256'].items():
        if sha(remember(Path(path), evidence)) != expected_sha:
            raise PermissionError('preserved previous lease changed')
    old_progress = Path(old['previous_progress_path'])
    if sha(remember(old_progress, evidence)) != old['previous_progress_sha256']:
        raise PermissionError('preserved previous progress changed')
    work = L / 'unified_runs' / P.name / RUNS[gpu]
    resource = strict(remember(work / 'resource_evidence.json', evidence))
    if any(resource.get(k) != progress[k] for k in ('pid', 'start_ticks', 'worker', 'physical_gpu_index', 'assignment_sha256', 'queue_sha256')):
        raise PermissionError('original run resource identity mismatch')
    return progress, work

def verify_claim(claim, receipt, progress, gpu, assignment_sha):
    row = receipt['source_row']
    expected = {'edition': P.name, 'worker': WORKER, 'gpu': gpu, 'assignment_sha256': assignment_sha,
        'assignment_index': receipt['assignment_index'], 'source_row': row, 'split': row['split'], 'id': row['id'],
        'pid': progress['pid'], 'owner_pid': progress['pid'], 'start_ticks': progress['start_ticks'],
        'run_id': progress['run_id'], 'automatic_retry': False}
    if any(claim.get(k) != v for k, v in expected.items()):
        raise PermissionError('original claim owner/PID/start/run/source/index mismatch')
    inactive(claim)

def verify_original_receipt(payloads, relative, receipt, row):
    manifest = strict(payloads[relative + '/fullframe_manifest.json'])
    beats = strict(payloads[relative + '/beats.json'])
    draft = strict(payloads[relative + '/draft.json'])
    copy = receipt['copy_evidence']
    if copy.get('mode') != 'direct_core_copy_source_one_pass' or copy.get('original_paths') != {k: row[k] for k in ('video_path', 'audio_path')}:
        raise ValueError('original receipt source copy provenance mismatch')
    hashes = copy['input_sha256']
    if hashes['video_path'] != manifest['video_sha256'] or hashes['audio_path'] != beats['audio_sha256']:
        raise ValueError('original receipt input hashes differ from outputs')
    if any(not re.fullmatch('[0-9a-f]{64}', hashes[k]) for k in ('video_path', 'audio_path')) or type(copy.get('copied_bytes')) is not int or copy['copied_bytes'] <= 0:
        raise ValueError('original receipt copy SHA/length malformed')
    if manifest['metadata_provenance']['input_source_row'] != row or receipt.get('schema') != 'ms_unified_novlm_case_receipt_v1' or receipt.get('status') != 'extracted_rule_draft_unapproved':
        raise ValueError('original receipt fullframe source/status mismatch')
    if receipt.get('draft_status') != draft['status'] or receipt.get('event_count') != len(draft['events']) or receipt.get('generated_with_action_events') != bool(draft['events']):
        raise ValueError('original receipt draft/event mismatch')
    if any(receipt.get(k) is not v for k, v in {'input_copy_completed': True, 'vlm_called': False, 'rgb_or_crops_written': False, 'beat_alignment_claimed': False}.items()):
        raise ValueError('original receipt no-inference/unapproved provenance mismatch')
    output_manifest = receipt.get('output_files')
    if output_manifest is not None:
        for name in ('pose.npz', 'fullframe_manifest.json', 'beats.json', 'draft.json'):
            entry = output_manifest[name]
            raw = payloads[relative + '/' + name]
            if entry['sha256'] != sha(raw) or entry['bytes'] != len(raw):
                raise ValueError('original receipt output SHA/length mismatch')
    return output_manifest is not None

def build_archive(local, payloads, files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as pack:
        for case in sorted((local / 'cases').glob('*/*')):
            for name in files:
                path = case / name
                relative = str(path.relative_to(local))
                info = pack.gettarinfo(str(path), arcname=relative)
                if not info.isfile() or info.size != len(payloads[relative]):
                    raise ValueError('candidate archive original file identity changed')
                pack.addfile(info, io.BytesIO(payloads[relative]))
    return buffer.getvalue()

def archive_contract(raw, payloads):
    if len(raw) % 10240 or raw[-1024:] != bytes(1024):
        raise ValueError('archive lacks complete tar block/footer')
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:') as pack:
        for member in pack:
            if not member.isfile() or member.name in seen or member.name not in payloads or member.size > MAX_FILE:
                raise ValueError('unsafe/duplicate/unexpected archive member')
            content = pack.extractfile(member).read()
            if member.size != len(content) or content != payloads[member.name]:
                raise ValueError('archive member SHA/length/bytes mismatch')
            seen.add(member.name)
    if seen != set(payloads):
        raise ValueError('archive missing full five-files/case member set')
    return len(seen)

def partial_archive_report(raw, payloads):
    seen = set()
    error = None
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode='r:') as pack:
            for member in pack:
                if not member.isfile() or member.name in seen or member.name not in payloads:
                    raise ValueError('unsafe/duplicate/unexpected original archive member')
                try:
                    content = pack.extractfile(member).read()
                except tarfile.ReadError as exc:
                    error = str(exc)
                    break
                if content != payloads[member.name] or len(content) != member.size:
                    raise ValueError('original archive completed member differs from original case')
                seen.add(member.name)
    except tarfile.ReadError as exc:
        error = str(exc)
    return {'bytes': len(raw), 'sha256': sha(raw), 'complete_members_verified': len(seen),
        'complete_archive': len(seen) == len(payloads) and len(raw) % 10240 == 0 and raw[-1024:] == bytes(1024),
        'read_error': error, 'original_tar_untouched': True}

def target_snapshot(target):
    anchor(target)
    if not target.exists():
        return {}
    result = {}
    for path in sorted(target.rglob('*')):
        anchor(path)
        if path.is_file():
            _, result[str(path.relative_to(target))] = read_evidence(path, MAX_ARCHIVE if path.name == 'batch.tar' else MAX_FILE)
        elif not path.is_dir():
            raise ValueError('unexpected target special file')
    return result

def verify_shared(target, payloads, archive):
    state = target_snapshot(target)
    for relative, entry in state.items():
        if relative == 'commit.json':
            continue
        expected = archive if relative == 'batch.tar' else payloads.get(relative)
        if expected is None or entry['sha256'] != sha(expected) or entry['bytes'] != len(expected):
            raise ValueError('existing shared output differs or unexpected: ' + relative)
        if relative == 'batch.tar':
            archive_contract(read_evidence(target / relative, MAX_ARCHIVE)[0], payloads)
    return state

def commit_index(excluded, evidence, selected):
    root = P / 'results' / WORKER / 'batches'
    paths = sorted(root.glob('gpu*/*/commit.json'))
    if len(paths) > 1100:
        raise ValueError('owned commit metadata exceeds finite expected budget')
    for path in paths:
        if path in excluded:
            continue
        commit = strict(remember(path, evidence))
        if commit.get('worker') != WORKER or commit.get('edition') != P.name:
            raise ValueError('owned commit edition/worker mismatch')
        for receipt in commit['receipts']:
            if receipt.get('status') == 'existing_claim_not_reassigned':
                continue
            row = receipt['source_row']
            if (row['split'].casefold(), row['id'].casefold()) in selected:
                raise ValueError('recovered source already committed elsewhere')
    return [str(x) for x in paths]

def validate_existing_commit(target, plan, payloads, archive, rules, validate):
    raw, _ = read_evidence(target / 'commit.json')
    commit = strict(raw)
    expected = {'schema': 'ms_unified_novlm_batch_commit_v1', 'edition': P.name, 'worker': WORKER,
        'gpu': plan['gpu'], 'run_id': plan['run_id'], 'assignment_sha256': plan['assignment_sha256'],
        'code_signatures': plan['code_signatures'], 'assignment_indices': plan['assignment_indices'],
        'receipts': plan['receipts'], 'batch_sha256': sha(archive), 'validation_first3': False,
        'quality_approved': False, 'unique_sources_attempted': len(plan['receipts']),
        'unique_sources_with_action_events': sum(bool(r['generated_with_action_events']) for r in plan['receipts']),
        'frame_count': sum(r['frame_count'] for r in plan['receipts']), **rules.FLAGS}
    if any(commit.get(k) != v for k, v in expected.items()):
        raise ValueError('existing shared commit original schema/provenance/counts mismatch')
    fields = {}
    for item in commit['publications']:
        relative = str(Path(item['path']).relative_to(target))
        content = archive if relative == 'batch.tar' else payloads.get(relative)
        if content is None or relative in fields or item['sha256'] != sha(content) or item['receipt'] != {'path': item['path'], 'bytes': len(content), 'sha256': sha(content)}:
            raise ValueError('existing commit publication receipt mismatch')
        actual = read_evidence(Path(item['path']), MAX_ARCHIVE if relative == 'batch.tar' else MAX_FILE)[0]
        if actual != content:
            raise ValueError('existing committed bytes mismatch')
        fields[relative] = item
    if set(fields) != set(payloads) | {'batch.tar'}:
        raise ValueError('existing committed full publication set mismatch')
    archive_contract(archive, payloads)
    for receipt in commit['receipts']:
        validate(payloads, target, receipt, receipt['source_row'], rules, strict)
    return {'status': 'already_committed_independently_deep_verified', 'commit_sha256': sha(raw), 'double_accounted': False}

def prepare(gpu, context):
    args, access, ticket, rows, assignment_sha, signatures, rules, files, validate, common = context
    evidence = dict(common)
    progress, work = original_gate(gpu, assignment_sha, signatures, args.queue_sha, evidence)
    name = RUNS[gpu] + '-batch176'
    local = work / name
    target = Path(ticket['output_root']) / 'batches' / f'gpu{gpu}' / name
    cases = sorted((local / 'cases').glob('*/*'))
    if len(cases) != 20:
        raise ValueError('exactly20 original complete cases required')
    payloads, receipts, deep, selected = {}, [], [], set()
    output_hashes = []
    for case in cases:
        if {p.name for p in case.iterdir()} != set(files):
            raise ValueError('original case missing/unexpected output file')
        for name in files:
            relative = str((case / name).relative_to(local))
            payloads[relative] = remember(case / name, evidence)
        relative = str(case.relative_to(local))
        receipt = strict(payloads[relative + '/receipt.json'])
        index = receipt['assignment_index']
        if type(index) is not int or index not in range(14080 + gpu, 14160, 4) or receipt['source_row'] != rows[index] or relative != 'cases/' + rows[index]['split'] + '/' + rows[index]['id']:
            raise ValueError('original assignment index/lane/case/source row mismatch')
        identity = (rows[index]['split'].casefold(), rows[index]['id'].casefold())
        if identity in selected:
            raise ValueError('duplicate recovered source')
        selected.add(identity)
        claim_path = Path(ticket['claims_root']) / rows[index]['split'] / (rows[index]['id'] + '.json')
        verify_claim(strict(remember(claim_path, evidence)), receipt, progress, gpu, assignment_sha)
        output_hashes.append(verify_original_receipt(payloads, relative, receipt, rows[index]))
        deep.append(validate(payloads, target, receipt, rows[index], rules, strict))
        receipts.append(receipt)
    receipts.sort(key=lambda x: x['assignment_index'])
    if [r['assignment_index'] for r in receipts] != list(range(14080 + gpu, 14160, 4)):
        raise ValueError('original20 contiguous lane receipts differ')
    no_run_processes()
    original_tar = local / 'batch.tar'
    original = partial_archive_report(remember(original_tar, evidence, MAX_ARCHIVE), payloads) if original_tar.exists() else {'exists': False, 'original_tar_untouched': True}
    archive = build_archive(local, payloads, files)
    members = archive_contract(archive, payloads)
    archive_path = CANDIDATE / name / 'batch.tar'
    write_once(archive_path, archive)
    state = verify_shared(target, payloads, archive)
    excluded = {Path(ticket['output_root']) / 'batches' / f'gpu{i}' / (RUNS[i] + '-batch176') / 'commit.json' for i in range(4)}
    index_paths = commit_index(excluded, evidence, selected)
    recheck(evidence)
    plan = {'schema': 'ms_publication_only_original_cache_candidate_v1', 'gpu': gpu, 'worker': WORKER,
        'edition': P.name, 'run_id': RUNS[gpu], 'batch_number': 176, 'local_original_batch': str(local),
        'target_batch': str(target), 'candidate_archive': str(archive_path), 'candidate_archive_sha256': sha(archive),
        'candidate_archive_bytes': len(archive), 'archive_members': members, 'core_files_order': list(files),
        'assignment_sha256': assignment_sha, 'queue_sha256': args.queue_sha, 'code_signatures': signatures,
        'assignment_indices': [r['assignment_index'] for r in receipts], 'receipts': receipts,
        'original_receipt_output_hash_manifest_present': all(output_hashes),
        'original_receipt_hash_limitation': 'Original receipts bind source input hashes/counts/flags; they do not contain a historical five-output SHA/length manifest. Candidate captures real original output SHA/bytes and verifies complete original partial-tar members and existing shared bytes; no historical SHA is fabricated.',
        'original_archive': original, 'files': {n: {'sha256': sha(b), 'bytes': len(b)} for n, b in payloads.items()},
        'evidence': evidence, 'original_commit_index_paths': index_paths, 'shared_target_snapshot': state,
        'deep_audit': deep, 'publication_only': True, 'models_loaded': False, 'inference_performed': False,
        'claims_modified': False, 'original_run_modified': False, 'shared_written': False, **rules.FLAGS}
    already = None
    if 'commit.json' in state:
        already = validate_existing_commit(target, plan, payloads, archive, rules, validate)
    return plan, payloads, archive, already

def prepublish(plan, payloads, archive, context):
    _, _, _, _, _, _, rules, _, validate, _ = context
    recheck(plan['evidence'])
    no_run_processes()
    if target_snapshot(Path(plan['target_batch'])) != plan['shared_target_snapshot']:
        raise PermissionError('shared target changed after frozen candidate; no publication')
    if read_evidence(Path(plan['candidate_archive']), MAX_ARCHIVE)[0] != archive:
        raise PermissionError('candidate archive changed')
    for receipt in plan['receipts']:
        row = receipt['source_row']
        relative = 'cases/' + row['split'] + '/' + row['id']
        for name in plan['core_files_order']:
            actual = read_evidence(Path(plan['local_original_batch']) / relative / name)[0]
            if actual != payloads[relative + '/' + name]:
                raise PermissionError('original case SHA/length changed')
        validate(payloads, Path(plan['target_batch']), receipt, row, rules, strict)
    excluded = {P / 'results' / WORKER / 'batches' / f'gpu{i}' / (RUNS[i] + '-batch176') / 'commit.json' for i in range(4)}
    selected = {(r['source_row']['split'].casefold(), r['source_row']['id'].casefold()) for r in plan['receipts']}
    paths = commit_index(excluded, {}, selected)
    if paths != plan['original_commit_index_paths']:
        raise PermissionError('original commit index changed')

def publish(plan, payloads, archive, context):
    _, access, _, _, _, _, rules, _, validate, _ = context
    if not plan['original_receipt_output_hash_manifest_present']:
        raise PermissionError('historical complete output SHA/length receipt absent; publication requires reviewed evidence-bound candidate authorization')
    prepublish(plan, payloads, archive, context)
    target = Path(plan['target_batch'])
    started = time.perf_counter()
    attempt = CANDIDATE / (plan['run_id'] + '-batch176') / 'publish_attempt.json'
    if attempt.exists():
        raise PermissionError('publication attempt already recorded; no automatic rerun')
    write_once(attempt, encoded({'candidate_sha256': sha(encoded(plan)), 'target_batch': str(target), 'started_unix': time.time(), 'publication_only': True}))
    publications = []
    for relative, content in payloads.items():
        recheck(plan['evidence'])
        receipt = access.publish_file(str(Path(plan['local_original_batch']) / relative), str(target / relative), sha(content))
        if access.read(str(target / relative), expected_sha=sha(content)) != content:
            raise ValueError('case publication readback mismatch; no retry')
        publications.append({'path': str(target / relative), 'sha256': sha(content), 'receipt': receipt})
    archive_receipt = access.publish_file(plan['candidate_archive'], str(target / 'batch.tar'), sha(archive))
    if access.read(str(target / 'batch.tar'), expected_sha=sha(archive)) != archive:
        raise ValueError('archive publication readback mismatch; no retry')
    publications.append({'path': str(target / 'batch.tar'), 'sha256': sha(archive), 'receipt': archive_receipt})
    recheck(plan['evidence'])
    no_run_processes()
    verify_shared(target, payloads, archive)
    commit = {'schema': 'ms_unified_novlm_batch_commit_v1', 'edition': P.name, 'worker': WORKER,
        'gpu': plan['gpu'], 'run_id': plan['run_id'], 'assignment_sha256': plan['assignment_sha256'],
        'code_signatures': plan['code_signatures'], 'assignment_indices': plan['assignment_indices'],
        'receipts': plan['receipts'], 'batch_sha256': sha(archive), 'publications': publications,
        'validation_first3': False, 'quality_approved': False, 'unique_sources_attempted': len(plan['receipts']),
        'unique_sources_with_action_events': sum(bool(r['generated_with_action_events']) for r in plan['receipts']),
        'frame_count': sum(r['frame_count'] for r in plan['receipts']), 'profile_seconds': {'pack_publish': time.perf_counter() - started},
        'publication_only': True, 'inference_performed': False, 'recovery_candidate_sha256': sha(encoded(plan)), **rules.FLAGS}
    access.publish_json(commit, str(target / 'commit.json'))
    if access.read(str(target / 'commit.json'), expected_sha=sha(encoded(commit))) != encoded(commit):
        raise ValueError('commit-last readback mismatch; no retry')
    result = validate_existing_commit(target, plan, payloads, archive, rules, validate)
    result['status'] = 'publication_only_committed_verified_no_original_progress_accounting'
    write_once(attempt.parent / 'publish_result.json', encoded(result))
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check-only', action='store_true')
    group.add_argument('--publish', action='store_true')
    parser.add_argument('--gpu', type=int, choices=range(4), action='append')
    parser.add_argument('--candidate-sha256', action='append', default=[])
    args = parser.parse_args(argv)
    context = setup()
    results = []
    prepared = []
    for gpu in args.gpu or range(4):
        plan, payloads, archive, already = prepare(gpu, context)
        plan_raw = encoded(plan)
        path = CANDIDATE / (plan['run_id'] + '-batch176') / 'candidate.json'
        if args.check_only:
            write_once(path, plan_raw)
        else:
            if not path.exists() or read_evidence(path)[0] != plan_raw or sha(plan_raw) not in args.candidate_sha256:
                raise PermissionError('explicit reviewed exact candidate SHA and unchanged write-once plan required')
        prepared.append((plan, payloads, archive, already))
        results.append({'gpu': gpu, 'candidate_path': str(path), 'candidate_sha256': sha(plan_raw),
            'source_count': len(plan['receipts']), 'archive_members': plan['archive_members'],
            'archive_sha256': sha(archive), 'archive_bytes': len(archive), 'original_archive': plan['original_archive'],
            'historical_output_hash_manifest_present': plan['original_receipt_output_hash_manifest_present'],
            'status': already['status'] if already else 'check_only_deep_verified_publication_gate_requires_historical_receipt_or_reviewed_evidence_authorization'})
    # All selected batches pass the complete contracts before the first shared write.
    if args.publish:
        for (plan, payloads, archive, already), result in zip(prepared, results):
            if already is not None:
                result.update(already)
            else:
                result.update(publish(plan, payloads, archive, context))
    print(json.dumps({'publication_only': True, 'shared_written': bool(args.publish), 'models_loaded': False,
        'claims_modified': False, 'original_run_modified': False, 'results': results}, sort_keys=True))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
