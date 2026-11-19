"""Isolated test-only small-case single-RPC publication; never a worker deployment."""
COMMON_CODE = r'''
import hashlib, json, os, re, stat, tempfile
from pathlib import Path
P = Path("/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/external_ms_unified_clusters80k_20261027_v1")
PREVIOUS_P = P.parent / "external_ms_unified_run_20261027_v1"
WORKERS = tuple("a100-extra-0" + str(n) for n in (3, 4, 5, 6))
SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")
FILES = {"pose.npz", "fullframe_manifest.json", "beats.json", "draft.json", "receipt.json"}
KEYS = {"id", "split", "video_path", "audio_path", "meta_path", "source_ordinal"}

def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def bad(value): raise ValueError("nonfinite JSON")
    if b"\0" in raw: raise ValueError("NUL in JSON; original bytes retained")
    return json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=pairs, parse_constant=bad)
def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()
def canonical(value):
    path = Path(value)
    if not isinstance(value, str) or not path.is_absolute() or str(path) != value or ".." in path.parts or "\0" in value:
        raise ValueError("noncanonical path")
    return path
def anchor(path):
    current = path
    while not current.exists() and not current.is_symlink(): current = current.parent
    if current.resolve(strict=True) != current: raise ValueError("redirected ancestor")
    return path
def sync(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
def mkdir(path, scope):
    if path != scope and scope not in path.parents: raise ValueError("directory outside scope")
    anchor(path); path.mkdir(parents=True, exist_ok=True); anchor(path); sync(path.parent)
def metadata(worker):
    if worker not in WORKERS: raise ValueError("worker not allowed")
    anchor(P)
    t, m = P / "assignments" / (worker + ".ticket.json"), P / "assignments" / (worker + ".jsonl")
    return t, m
def queue_path(worker):
    metadata(worker)
    return P / (worker + ".queue.json")
def validate_queue(worker, qraw, tsha, qsha=None):
    queue = strict(qraw); tpath, _ = metadata(worker)
    if (not isinstance(queue, dict) or queue.get("schema") != "ms_external_authorized_rollover_queue_v1"
        or queue.get("worker_id") != worker
        or queue.get("automatic_rollover_within_preallocated_tickets") is not True
        or queue.get("no_automatic_retry_sealed_previous_sources") is not True
        or queue.get("no_restart_active_previous_job") is not True
        or queue.get("run_until_preallocated_queue_exhausted") is not True
        or (qsha is not None and hashlib.sha256(qraw).hexdigest() != qsha)):
        raise ValueError("owned finite rollover queue binding")
    refs = queue.get("tickets")
    paths = (PREVIOUS_P / "assignments" / (worker + ".ticket.json"), tpath)
    if not isinstance(refs, list) or len(refs) != 2: raise ValueError("finite queue must contain exactly two ticket refs")
    for ref, path in zip(refs, paths):
        if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"} or ref.get("path") != str(path)
            or not isinstance(ref.get("sha256"), str) or not SHA.fullmatch(ref["sha256"])):
            raise ValueError("queue ownworker old/new ticket reference")
    if refs[1]["sha256"] != tsha: raise ValueError("queue new ticket exact SHA")
    return queue
def validate(worker, traw, mraw, tsha, msha):
    ticket = strict(traw); tpath, mpath = metadata(worker)
    if (not isinstance(ticket, dict) or ticket.get("schema") != "ms_external_unified_assignment_v1"
        or ticket.get("worker_id") != worker or ticket.get("edition") != P.name
        or type(ticket.get("source_count")) is not int or ticket["source_count"] != 20000
        or ticket.get("assignment_path") != str(mpath) or ticket.get("claims_root") != str(P / "claims")
        or ticket.get("output_root") != str(P / "results" / worker)
        or hashlib.sha256(traw).hexdigest() != tsha or hashlib.sha256(mraw).hexdigest() != msha
        or ticket.get("assignment_sha256") != msha): raise ValueError("owned ticket/assignment SHA or roots mismatch")
    refs = ticket.get("code_refs")
    if not isinstance(refs, list) or not refs: raise ValueError("missing code_refs")
    for ref in refs:
        if not isinstance(ref, dict) or not SHA.fullmatch(str(ref.get("sha256", ""))): raise ValueError("code_ref SHA")
        canonical(ref["path"])
    if not mraw or b"\0" in mraw or not mraw.endswith(b"\n"): raise ValueError("invalid complete JSONL; original bytes retained")
    rows, identities, ordinals = [], set(), set()
    for line in mraw.split(b"\n")[:-1]:
        row = strict(line)
        if not isinstance(row, dict) or set(row) != KEYS: raise ValueError("row schema")
        if any(not isinstance(row[k], str) or not SAFE.fullmatch(row[k]) for k in ("split", "id")): raise ValueError("unsafe identity")
        identity = (row["split"].casefold(), row["id"].casefold())
        ordinal = row["source_ordinal"]
        if identity in identities or type(ordinal) is not int or ordinal < 1 or ordinal in ordinals: raise ValueError("duplicate identity/ordinal")
        identities.add(identity); ordinals.add(ordinal)
        for key in ("video_path", "audio_path", "meta_path"): canonical(row[key])
        rows.append(row)
    if len(rows) != 20000: raise ValueError("assignment not exactly 20000")
    return ticket, rows
'''
exec(COMMON_CODE, globals())
EXPERIMENT_CODE = r'''
import argparse, base64, io, signal, sys, tarfile, time, subprocess, shlex
sys.dont_write_bytecode = True
RUN = 'batchrpc_20261027_v1_01'
LOCAL_ROOT = Path('/tmp/ms_pose2d_draft_20261026T1427Z/worker-0/publication_experiments') / RUN
OUTPUT_ROOT = P / 'results/a100-extra-03/publication_experiments' / RUN
SAMPLE_ROOT = P / 'results/a100-extra-03/batches/gpu0/g0-1793043643552659367-0bb56122-batch007'
WORKER, EXECUTION, GPU = 'a100-extra-03', 'a100-extra-04', 0
INDICES = [560, 564, 568, 572]
MAX_BYTES, ORIGINAL_CAP = 8 * 2**20, 32 * 2**20
TICKET_SHA = '5a20df8e2975d506d4890359a503dd6e821331d587fe0910538244835b4957eb'
META_SHA = '8919d280bc188daf389ac8cf27cfc6b54184f6d7f82d2ab2e7848c91512b9788'
QUEUE_SHA = '392e05e397d6fb387b1b76967c184d93bb19b553e735b83c19b75fc7ca5c0c25'
COMMIT_SHA = '366eea2afe833fb35319c8e5d3b050ecd4d83eee384dffadf67d3891f3d49666'
ORIGINAL_ARCHIVE_SHA = 'fe9547f12a27f3d0d71e0c902f7eb847fb62eb4acc20b632d303582c05430d2e'

def ex_sha(raw): return hashlib.sha256(raw).hexdigest()

def ex_read(path, cap, expected):
    anchor(path)
    with path.open('rb') as f:
        before = os.fstat(f.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > cap: raise ValueError('opened FD regular/byte cap')
        raw = f.read(cap + 1); after = os.fstat(f.fileno())
    if len(raw) != before.st_size or after.st_size != before.st_size or ex_sha(raw) != expected:
        raise ValueError('opened FD read length/content expected SHA; no ctime assertion')
    return raw

def ex_identity(indices, worker, node, gpu):
    if worker != WORKER or node != EXECUTION or gpu != GPU or indices != INDICES:
        raise ValueError('test assignment worker03/execution04/GPU0/exact sample indices')
    if any(type(i) is not int or i % 4 != GPU for i in indices): raise ValueError('original assignment index modulo4')

def ex_member_name(name):
    parts = name.split('/')
    if len(parts) != 4 or parts[0] != 'cases' or not all(SAFE.fullmatch(v) for v in parts[1:3]) or parts[3] not in FILES:
        raise ValueError('member canonical original case path/basename')
    return name

def ex_tar(raw, mapping):
    if len(raw) > MAX_BYTES or not mapping or len(mapping) > 100: raise ValueError('tar/member byte budget')
    found, total = {}, 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:') as pack:
        for member in pack:
            name = ex_member_name(member.name)
            if not member.isreg() or member.name in found or member.linkname or set(member.pax_headers) - {'mtime', 'atime', 'ctime'}:
                raise ValueError('tar duplicate/symlink/hardlink/directory/PAX')
            if name not in mapping or member.size != mapping[name]['bytes']: raise ValueError('unexpected member/size')
            total += member.size
            if total > MAX_BYTES: raise ValueError('expanded byte budget')
            f = pack.extractfile(member); payload = f.read(member.size + 1)
            if len(payload) != member.size or ex_sha(payload) != mapping[name]['sha256']: raise ValueError('member expected SHA/readlen')
            found[name] = payload
    if set(found) != set(mapping): raise ValueError('exact complete member allowlist')
    return found

def ex_pack(payloads):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:', format=tarfile.USTAR_FORMAT) as pack:
        for name in sorted(payloads):
            info = tarfile.TarInfo(ex_member_name(name)); info.size = len(payloads[name]); info.mode = 0o600; info.mtime = 0
            pack.addfile(info, io.BytesIO(payloads[name]))
    raw = stream.getvalue()
    if len(raw) > MAX_BYTES: raise ValueError('packed byte budget')
    return raw

def ex_proof():
    tpath, mpath = metadata(WORKER)
    traw = ex_read(tpath, 2**20, TICKET_SHA)
    mraw = ex_read(mpath, 32*2**20, META_SHA)
    qraw = ex_read(queue_path(WORKER), 2**20, QUEUE_SHA)
    validate_queue(WORKER, qraw, TICKET_SHA, QUEUE_SHA)
    _, rows = validate(WORKER, traw, mraw, TICKET_SHA, META_SHA)
    craw = ex_read(SAMPLE_ROOT / 'commit.json', 2**20, COMMIT_SHA); c = strict(craw)
    if (c.get('schema') != 'ms_unified_novlm_batch_commit_v1' or c.get('worker') != WORKER or c.get('gpu') != GPU
        or c.get('edition') != P.name or c.get('assignment_sha256') != META_SHA
        or c.get('batch_sha256') != ORIGINAL_ARCHIVE_SHA or len(c.get('receipts', [])) != 20):
        raise ValueError('original committed sample identity')
    if c.get('assignment_indices') != [r.get('assignment_index') for r in c['receipts']]: raise ValueError('sample source index evidence')
    for r in c['receipts']:
        i = r.get('assignment_index')
        if type(i) is not int or not 0 <= i < len(rows) or i % 4 != GPU or r.get('source_row') != rows[i]:
            raise ValueError('sample member of own03 exact metadata/GPU0 row')
    receipts = c['receipts'][:4]
    ex_identity([r['assignment_index'] for r in receipts], WORKER, EXECUTION, GPU)
    allmap = {}
    for item in c['publications']:
        path = canonical(item['path']); relative = str(path.relative_to(SAMPLE_ROOT)); rec = item['receipt']
        if (relative in allmap or rec.get('path') != str(path) or rec.get('sha256') != item.get('sha256')
            or not SHA.fullmatch(str(item.get('sha256'))) or type(rec.get('bytes')) is not int or not 0 <= rec['bytes'] <= MAX_BYTES):
            raise ValueError('original publication expected SHA/receipt')
        if relative == 'batch.tar':
            if item['sha256'] != ORIGINAL_ARCHIVE_SHA or rec['bytes'] != 3891200: raise ValueError('original archive witness')
        else:
            ex_member_name(relative); allmap[relative] = dict(bytes=rec['bytes'], sha256=item['sha256'])
    if len(allmap) != 100: raise ValueError('original exact100 files')
    fullnames = {'cases/' + r['source_row']['split'] + '/' + r['source_row']['id'] + '/' + name for r in c['receipts'] for name in FILES}
    if set(allmap) != fullnames: raise ValueError('original20 sources five files allowlist')
    selected = {'cases/' + r['source_row']['split'] + '/' + r['source_row']['id'] + '/' + name for r in receipts for name in FILES}
    proof = dict(test_only=True, no_annotation=True, newsource=0, assignment_worker=WORKER, execution_node=EXECUTION,
                 gpu=GPU, assignment_indices=INDICES, ticket_sha256=TICKET_SHA, assignment_sha256=META_SHA,
                 queue_sha256=QUEUE_SHA, original_commit_path=str(SAMPLE_ROOT/'commit.json'), original_commit_sha256=COMMIT_SHA,
                 original_archive_sha256=ORIGINAL_ARCHIVE_SHA, original_archive_bytes=3891200,
                 original_sources=20, original_case_files=100, receipts=receipts,
                 members={name:allmap[name] for name in sorted(selected)})
    return proof, allmap

def ex_root(arm):
    if arm not in ('baseline', 'batch'): raise ValueError('arm allowlist')
    return OUTPUT_ROOT / arm / 'gpu0' / SAMPLE_ROOT.name

def ex_commit(proof, arm, archive_sha, archive_bytes):
    if not SHA.fullmatch(str(archive_sha)) or type(archive_bytes) is not int or not 0 < archive_bytes <= MAX_BYTES:
        raise ValueError('experimental archive SHA/size')
    target = ex_root(arm)
    pubs = [dict(path=str(target/name), **proof['members'][name]) for name in sorted(proof['members'])]
    pubs.append(dict(path=str(target/'batch.tar'), bytes=archive_bytes, sha256=archive_sha))
    return dict(schema='ms_test_only_batch_rpc_commit_v1', run_id=RUN, output_prefix=str(target),
                archive_sha256=archive_sha, archive_bytes=archive_bytes, publications=pubs, expanded_files=20, **proof)

def ex_write(path, raw, expected, root):
    if path != root and root not in path.parents: raise ValueError('write outside fixed experimental root')
    if path == root or path.name.startswith('.'): raise ValueError('invalid output path')
    anchor(path); mkdir(path.parent, root); anchor(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.batchrpc-', delete=False) as f:
            temporary = Path(f.name); f.write(raw); f.flush(); os.fsync(f.fileno())
        if ex_sha(raw) != expected: raise ValueError('publish payload expectedSHA')
        try: os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            print('batchrpc warning: write-once existing path retained ' + str(path), file=sys.stderr, flush=True)
        sync(path.parent)
        ex_read(path, MAX_BYTES, expected)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)

def ex_request_check(q):
    ex_identity(q.get('assignment_indices'), q.get('assignment_worker'), q.get('execution_node'), q.get('gpu'))
    if q.get('test_only') is not True or q.get('no_annotation') is not True or q.get('newsource') != 0 or q.get('run_id') != RUN:
        raise ValueError('testmode gate')
    if set(q) - {'op','arm','name','size','sha256','archive_sha256','archive_bytes','commit_sha256',
                  'assignment_indices','assignment_worker','execution_node','gpu','test_only','no_annotation','newsource','run_id'}:
        raise ValueError('arbitrary path/options forbidden')
    if q.get('op') not in ('prepare','file','batch','readback'): raise ValueError('operation forbidden')
    if q['op'] != 'prepare': ex_root(q.get('arm'))
    if q['op'] in ('file','batch'):
        if type(q.get('size')) is not int or not 0 <= q['size'] <= MAX_BYTES or not SHA.fullmatch(str(q.get('sha256'))):
            raise ValueError('RPC payload budget/SHA')
        if q['op'] == 'file' and q.get('name') not in ('batch.tar','commit.json'): ex_member_name(q.get('name',''))
        if q['op'] == 'batch' and q.get('arm') != 'batch': raise ValueError('batch arm binding')

def ex_receiver(q, incoming):
    ex_request_check(q)
    proof, allmap = ex_proof()
    if q['op'] == 'prepare':
        if incoming.read(1): raise ValueError('unexpected prepare payload')
        original = ex_read(SAMPLE_ROOT/'batch.tar', ORIGINAL_CAP, ORIGINAL_ARCHIVE_SHA)
        payloads = ex_tar(original, allmap)
        selected = {name:payloads[name] for name in proof['members']}
        archive = ex_pack(selected)
        return encoded(dict(proof=proof, archive_sha256=ex_sha(archive), archive_bytes=len(archive))) + archive
    root = ex_root(q['arm'])
    c = ex_commit(proof, q['arm'], q['archive_sha256'], q['archive_bytes']); craw = encoded(c)
    if ex_sha(craw) != q.get('commit_sha256'): raise ValueError('commit expected SHA binding')
    if q['op'] == 'readback':
        if incoming.read(1): raise ValueError('unexpected readback payload')
        raw = ex_read(root/'commit.json', MAX_BYTES, q['commit_sha256'])
        if raw != craw: raise ValueError('exact testcommit bytes')
        for item in c['publications']: ex_read(Path(item['path']), MAX_BYTES, item['sha256'])
        return raw
    payload = incoming.read(q['size'] + 1)
    if len(payload) != q['size'] or ex_sha(payload) != q['sha256']: raise ValueError('payload expectedSHA/readlen')
    if q['op'] == 'batch' or q.get('name') == 'batch.tar':
        if q['sha256'] != q['archive_sha256'] or len(payload) != q['archive_bytes']: raise ValueError('archive expectedSHA/size')
        expanded = ex_tar(payload, proof['members'])
        if q['op'] == 'batch':
            for name in sorted(expanded): ex_write(root/name, expanded[name], proof['members'][name]['sha256'], OUTPUT_ROOT)
        ex_write(root/'batch.tar', payload, q['archive_sha256'], OUTPUT_ROOT)
    elif q['name'] == 'commit.json':
        if payload != craw: raise ValueError('baseline exact testcommit payload')
        for item in c['publications']: ex_read(Path(item['path']), MAX_BYTES, item['sha256'])
        ex_write(root/'commit.json', craw, q['commit_sha256'], OUTPUT_ROOT)
    else:
        expected = proof['members'].get(q['name'])
        if expected != dict(bytes=len(payload), sha256=q['sha256']): raise ValueError('sample member expected mapping')
        ex_write(root/q['name'], payload, q['sha256'], OUTPUT_ROOT)
    if q['op'] == 'batch':
        for item in c['publications']: ex_read(Path(item['path']), MAX_BYTES, item['sha256'])
        ex_write(root/'commit.json', craw, q['commit_sha256'], OUTPUT_ROOT)
    return encoded(dict(test_only=True, bytes=len(payload), sha256=q['sha256'], expanded_files=20 if q['op']=='batch' else None))

def ex_base(op, **fields):
    return dict(op=op, assignment_worker=WORKER, execution_node=EXECUTION, gpu=GPU, assignment_indices=INDICES,
                run_id=RUN, test_only=True, no_annotation=True, newsource=0, **fields)

def ex_rpc(q, payload=b''):
    code = "__name__ = 'batchrpc_receiver'\n" + COMMON_CODE + '\n' + EXPERIMENT_CODE + '\nex_remote_main()\n'
    command = 'python3 -c ' + shlex.quote(code)
    argv = ['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/ms_pilot_relay_known_hosts',
            '-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','11.239.36.34',command]
    remain = DEADLINE - time.monotonic()
    if remain <= 0: raise TimeoutError('experiment total180sec deadline')
    result = subprocess.run(argv, input=encoded(q)+payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=remain)
    if result.stderr: sys.stderr.buffer.write(result.stderr); sys.stderr.buffer.flush()
    if result.returncode: raise RuntimeError('receiver failed sealed, no retry: '+str(result.returncode))
    return result.stdout

def ex_remote_main():
    try:
        line = sys.stdin.buffer.readline(16385)
        if len(line) > 16384 or not line.endswith(b'\n'): raise ValueError('bounded RPC header')
        raw = ex_receiver(strict(line), sys.stdin.buffer)
        sys.stdout.buffer.write(raw); sys.stdout.buffer.flush()
    except Exception as error:
        print('batchrpc rejected: '+type(error).__name__+': '+str(error), file=sys.stderr, flush=True); sys.exit(1)

def ex_selftest():
    names = ['cases/train/example/'+name for name in sorted(FILES)]
    payloads = {name:b'fixture-'+name.encode() for name in names}
    mapping = {name:dict(bytes=len(raw), sha256=ex_sha(raw)) for name,raw in payloads.items()}
    archive = ex_pack(payloads)
    assert ex_tar(archive, mapping) == payloads
    checks = []
    def rejected(label, action):
        try: action()
        except (ValueError, TypeError, AttributeError, tarfile.TarError): checks.append(label)
        else: raise AssertionError('accepted '+label)
    for bad in ['../cases/train/example/pose.npz','cases/train/../pose.npz','/cases/train/example/pose.npz','cases/train/example/notallowed.json']:
        rejected('path:'+bad, lambda bad=bad: ex_member_name(bad))
    rejected('wrongGPUindex',lambda:ex_identity([560,564,568,573],WORKER,EXECUTION,GPU))
    rejected('impersonatedworker04',lambda:ex_identity(INDICES,EXECUTION,EXECUTION,GPU))
    rejected('arbitraryoutput',lambda:ex_request_check(ex_base('prepare',path='/tmp/foreign')))
    rejected('payloadbudget',lambda:ex_request_check(ex_base('batch',arm='batch',size=MAX_BYTES+1,sha256=ex_sha(b''))))
    rejected('memberSHA',lambda:ex_tar(archive,{name:dict(mapping[name],sha256='0'*64) for name in mapping}))
    rejected('archivebudget',lambda:ex_tar(b'0'*(MAX_BYTES+1),mapping))
    for label, kind in [('symlink',tarfile.SYMTYPE),('hardlink',tarfile.LNKTYPE),('directory',tarfile.DIRTYPE),('duplicate',tarfile.REGTYPE)]:
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w:',format=tarfile.USTAR_FORMAT) as pack:
            info=tarfile.TarInfo(names[0]);info.type=kind
            if kind==tarfile.REGTYPE:
                info.size=len(payloads[names[0]]);pack.addfile(info,io.BytesIO(payloads[names[0]]));pack.addfile(info,io.BytesIO(payloads[names[0]]))
            else: info.linkname='outside' if kind!=tarfile.DIRTYPE else '';pack.addfile(info)
        rejected(label,lambda:ex_tar(stream.getvalue(),mapping))
    return dict(test_only=True, positive_archive_members=5, negative_checks=checks, writes=0, models_loaded=0, media_reads=0)

def ex_run():
    global DEADLINE
    DEADLINE = time.monotonic()+180
    def timeout_handler(signum, frame):
        raise TimeoutError('experiment self timeout180sec; production untouched')
    signal.signal(signal.SIGALRM, timeout_handler); signal.alarm(180)
    started=time.perf_counter()
    prepared = ex_rpc(ex_base('prepare'))
    header, archive = prepared.split(b'\n',1); witness=strict(header); proof=witness['proof']
    if ex_sha(archive)!=witness['archive_sha256'] or len(archive)!=witness['archive_bytes']: raise ValueError('prepared archive expectedSHA/readlen')
    ex_identity(proof['assignment_indices'],proof['assignment_worker'],proof['execution_node'],proof['gpu'])
    members=ex_tar(archive,proof['members'])
    if len(members)!=20 or len(proof['receipts'])!=4: raise ValueError('exact4source20file sample')
    # Keep the selected payload/provenance as experiment artifacts, never modify original paths inside payloads.
    for name in sorted(members): ex_write(LOCAL_ROOT/'payload'/name,members[name],proof['members'][name]['sha256'],LOCAL_ROOT)
    ex_write(LOCAL_ROOT/'payload/batch.tar',archive,witness['archive_sha256'],LOCAL_ROOT)
    ex_write(LOCAL_ROOT/'original_source_evidence.json',encoded(witness),ex_sha(encoded(witness)),LOCAL_ROOT)
    result=dict(test_only=True,no_annotation=True,newsource=0,run_id=RUN,local_root=str(LOCAL_ROOT),shared_root=str(OUTPUT_ROOT),
                setup_rpc_calls=1,setup_wall_seconds=time.perf_counter()-started,archive_bytes=len(archive),
                archive_sha256=ex_sha(archive),expanded_payload_bytes=sum(map(len,members.values())),
                expanded_files_per_arm=20,regular_files_per_arm_including_tar_commit=22,single_copy_thread=True,
                ssh_topology='a100-extra-04 -> 11.239.36.34 (a100-extra-03), verified existing host key, default port36000',
                production_pipeline_speedup_proven=False,arms=[])
    for arm in ('baseline','batch'):
        c=ex_commit(proof,arm,ex_sha(archive),len(archive));craw=encoded(c)
        binding=dict(arm=arm,archive_sha256=ex_sha(archive),archive_bytes=len(archive),commit_sha256=ex_sha(craw))
        armstart=time.perf_counter();copystart=armstart;calls=0;sent=0
        items=[(name,members[name]) for name in sorted(members)]+[('batch.tar',archive),('commit.json',craw)] if arm=='baseline' else [('batch.tar',archive)]
        for name, raw in items:
            q=ex_base('file' if arm=='baseline' else 'batch',name=name,size=len(raw),sha256=ex_sha(raw),**binding)
            ack=strict(ex_rpc(q,raw));calls+=1;sent+=len(raw)
            if ack.get('bytes')!=len(raw) or ack.get('sha256')!=ex_sha(raw): raise ValueError('receiver payload receipt')
        copywall=time.perf_counter()-copystart
        readback=ex_rpc(ex_base('readback',**binding));calls+=1
        if readback!=craw or ex_sha(readback)!=binding['commit_sha256']: raise ValueError('commit readback expectedSHA')
        r=dict(arm=arm,complete=True,wall_seconds=time.perf_counter()-armstart,copy_seconds=copywall,
               publication_rpc_calls=calls-1,readback_rpc_calls=1,total_rpc_calls=calls,copy_payload_bytes=sent,
               expanded_files=20,regular_files=22,assignment_indices=INDICES,assignment_worker=WORKER,execution_node=EXECUTION,
               commit_path=str(ex_root(arm)/'commit.json'),commit_sha256=ex_sha(readback),archive_sha256=ex_sha(archive),
               all20member_sha_readback=True,commit_readback=True)
        result['arms'].append(r)
        ex_write(LOCAL_ROOT/(arm+'_result.json'),encoded(r),ex_sha(encoded(r)),LOCAL_ROOT)
        print(json.dumps(r),file=sys.stderr,flush=True)
    result['total_wall_seconds']=time.perf_counter()-started
    result['publication_contract_wall_ratio']=result['arms'][0]['wall_seconds']/result['arms'][1]['wall_seconds']
    ex_write(LOCAL_ROOT/'result.json',encoded(result),ex_sha(encoded(result)),LOCAL_ROOT)
    print(json.dumps(result),flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--selftest',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.selftest: print(json.dumps(ex_selftest()))
    elif args.run: ex_run()
    else: parser.error('explicit --selftest or --run required; no production mode')
'''
exec(EXPERIMENT_CODE, globals())
