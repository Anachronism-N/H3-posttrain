"""Finite preallocated rollover I/O; independent derivative of frozen transport v3."""
from __future__ import annotations
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import selectors
import stat as stat_module
import subprocess
import tempfile
import time

# The same validation/write-once code executes locally and on the fixed relay.
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
LIMIT = 8 * 2**30

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
def snapshot(s):
    return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns, device=s.st_dev, inode=s.st_ino)
def regular(path):
    anchor(path)
    s = path.stat()
    if not stat.S_ISREG(s.st_mode): raise ValueError("not regular")
    return snapshot(s)
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
def raw_read(path, limit=4*1024*1024):
    anchor(path)
    if not path.is_file() or path.stat().st_size > limit: raise ValueError("metadata size/type")
    raw = path.read_bytes()
    if len(raw) > limit: raise ValueError("metadata changed beyond bounded size")
    return raw
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

def owned_output(path, output, rows, directory=False):
    try: parts = path.relative_to(output).parts
    except ValueError: return False
    names = {"summary.json", "run_identity.json", "first3_verified.json", "commit.json", "resource_evidence.json", "resource_checks.json"}
    if not parts: return directory
    if len(parts) == 1: return parts[0] in names or bool(re.fullmatch(r"resource_[A-Za-z0-9_-]+\.json", parts[0])) or (directory and parts[0] in ("batches", "errors", "resource_evidence"))
    if parts[0] == "resource_evidence": return len(parts) == 2 and SAFE.fullmatch(parts[1][:-5]) and parts[1].endswith(".json")
    if parts[0] not in ("batches", "errors") or parts[1] not in ("gpu0", "gpu1", "gpu2", "gpu3"): return False
    gpu = int(parts[1][-1])
    if parts[0] == "errors" and len(parts) in (4, 5) and parts[-1].endswith(".json"):
        split, identifier = parts[-2], parts[-1][:-5]
        if len(parts) == 5 and not SAFE.fullmatch(parts[2]): return False
        return any(i % 4 == gpu and row["split"] == split and row["id"] == identifier for i, row in enumerate(rows))
    if len(parts) == 2: return directory
    if not SAFE.fullmatch(parts[2]): return False
    if len(parts) == 3: return directory
    if len(parts) == 4: return parts[3] in names | {"batch.tar", "batch.tar.gz", "error_receipt.json"} or (directory and parts[3] == "cases")
    if parts[3] != "cases": return False
    matches = [row for index, row in enumerate(rows) if index % 4 == gpu and row["split"] == parts[4] and (len(parts) == 5 or row["id"] == parts[5])]
    if not matches: return False
    if len(parts) in (5, 6): return directory
    return len(parts) == 7 and parts[6] in FILES | {"error_receipt.json"}
def dispatch(q, reader, writer):
    worker, op, path = q["worker"], q["op"], canonical(q["path"])
    tpath, mpath = metadata(worker); qpath = queue_path(worker)
    if op == "read" and path in (tpath, mpath, qpath) and not q.get("ticket_sha256"):
        return raw_read(path, 64*1024*1024 if path == mpath else 1024*1024)
    traw, mraw = raw_read(tpath, 1024*1024), raw_read(mpath, 64*1024*1024)
    qraw = raw_read(qpath, 1024*1024)
    qsha = q.get("queue_sha256")
    if not isinstance(qsha, str) or not SHA.fullmatch(qsha): raise ValueError("queue SHA binding required")
    validate_queue(worker, qraw, q.get("ticket_sha256"), qsha)
    ticket, rows = validate(worker, traw, mraw, q.get("ticket_sha256"), q.get("assignment_sha256"))
    if op == "read" and path in (tpath, mpath, qpath):
        return {tpath: traw, mpath: mraw, qpath: qraw}[path]
    media = {row[k] for row in rows for k in ("video_path", "audio_path", "meta_path")}
    output, claims = Path(ticket["output_root"]), Path(ticket["claims_root"])
    if op == "read" and owned_output(path, output, rows):
        before = regular(path)
        cap = 8 * 2**30 if path.name in ("batch.tar", "batch.tar.gz") else 64 * 2**20
        if before["bytes"] > cap:
            raise ValueError("owned output readback exceeds bounded limit")
        expected = q.get("expected_sha256")
        if not isinstance(expected, str) or not SHA.fullmatch(expected):
            raise ValueError("owned output readback requires published content SHA256")
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            raw = stream.read(cap + 1)
            after = os.fstat(stream.fileno())
        if len(raw) != opened.st_size or opened.st_size != after.st_size or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("owned output size or published SHA256 differs during readback")
        return raw
    claim_rows = {claims / row["split"] / (row["id"] + ".json"): (i, row) for i, row in enumerate(rows)}
    if op == "stat" and str(path) in media: return regular(path)
    if op == "read" and str(path) in {row["meta_path"] for row in rows}: return raw_read(path)
    if op == "exists" and (path == claims or path in claim_rows or path in {claims / row["split"] for row in rows} or owned_output(path, output, rows, True) or owned_output(path, output, rows)):
        anchor(path); return dict(exists=os.path.lexists(path))
    if op == "claim" and path in claim_rows:
        index, row = claim_rows[path]; record = q["record"]
        gpu = record.get("physical_gpu", record.get("gpu"))
        if (record.get("edition") != P.name or record.get("worker", record.get("worker_id")) != worker
            or type(gpu) is not int or gpu not in range(4) or index % 4 != gpu
            or type(record.get("pid")) is not int or record["pid"] <= 0
            or record.get("assignment_sha256") != q["assignment_sha256"]
            or type(record.get("assignment_index")) is not int or record["assignment_index"] != index
            or record.get("source_row") != row
            or any(record.get(k) != row[k] for k in ("split", "id"))
            or record.get("automatic_retry", False) is not False
            or ("worker_id" in record and record["worker_id"] != worker)
            or ("worker" in record and record["worker"] != worker)
            or ("gpu" in record and record["gpu"] != gpu)
            or ("source_ordinal" in record and record["source_ordinal"] != row["source_ordinal"])): raise ValueError("claim record binding")
        if "input_original" in record and record["input_original"] != {k: row[k] for k in ("video_path", "audio_path", "meta_path")}: raise ValueError("claim source binding")
        mkdir(path.parent, claims); anchor(path)
        try:
            with path.open("xb") as stream: stream.write(encoded(record)); stream.flush(); os.fsync(stream.fileno())
            sync(path.parent); return dict(created=True)
        except FileExistsError: return dict(created=False)
    if op == "copy" and str(path) in media:
        expected = q["snapshot"]; anchor(path)
        with path.open("rb") as stream:
            s = os.fstat(stream.fileno()); before = snapshot(s)
            if not stat.S_ISREG(s.st_mode) or before != expected or expected["bytes"] > q["max_bytes"]: raise ValueError("source before/budget")
            writer.write(encoded(dict(source_stat_before=before))); writer.flush(); copied = 0
            while copied < expected["bytes"]:
                chunk = stream.read(min(4*1024*1024, expected["bytes"]-copied))
                if not chunk: raise ValueError("short source")
                writer.write(chunk); copied += len(chunk)
            after = snapshot(os.fstat(stream.fileno()))
            if stream.read(1) or after != expected or regular(path) != expected: raise ValueError("source changed during stream")
        writer.write(encoded(dict(source_stat_after=after, copied=copied))); writer.flush(); return None
    if op == "publish" and owned_output(path, output, rows):
        size, expected = q["size"], q["sha256"]
        if type(size) is not int or not 0 <= size <= LIMIT or not isinstance(expected, str) or not SHA.fullmatch(expected): raise ValueError("publication size/SHA")
        mkdir(path.parent, output); anchor(path); temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".unified-", delete=False) as target:
                temporary = Path(target.name); h = hashlib.sha256(); copied = 0
                while copied < size:
                    chunk = reader.read(min(4*1024*1024, size-copied))
                    if not chunk: raise ValueError("short publication")
                    target.write(chunk); h.update(chunk); copied += len(chunk)
                if reader.read(1) or h.hexdigest() != expected: raise ValueError("publication SHA/length")
                target.flush(); os.fsync(target.fileno())
            try: os.link(temporary, path, follow_symlinks=False)
            except FileExistsError: pass
            sync(path.parent); anchor(path); h = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(4*1024*1024), b""): h.update(chunk)
            if regular(path)["bytes"] != size or h.hexdigest() != expected: raise ValueError("write-once existing/new output mismatch")
            return dict(path=str(path), bytes=size, sha256=expected)
        finally:
            if temporary is not None: temporary.unlink(missing_ok=True)
    raise ValueError("RPC operation/path not allowlisted")
'''
COMMON_CODE += r'''
import tarfile
BATCH_MAX_SOURCES = 20
BATCH_METADATA_CAP = 4 * 2**20
DIAGNOSTIC_FILES = {'beatthis_rejected.npz', 'beatthis_rejected.json'}
_old_owned_output = owned_output
_old_dispatch = dispatch

def owned_output(path, output, rows, directory=False):
    if _old_owned_output(path, output, rows, directory): return True
    try: parts = path.relative_to(output).parts
    except ValueError: return False
    if len(parts) != 7 or parts[0] != 'batches' or parts[1] not in ('gpu0','gpu1','gpu2','gpu3') or not SAFE.fullmatch(parts[2]) or parts[3] != 'diagnostics' or parts[6] not in DIAGNOSTIC_FILES: return False
    return any(i % 4 == int(parts[1][-1]) and row['split'] == parts[4] and row['id'] == parts[5] for i, row in enumerate(rows))

def hash_regular(path, size, expected):
    before = regular(path)
    if before['bytes'] != size: raise ValueError('existing member byte conflict')
    h = hashlib.sha256(); count = 0
    with path.open('rb') as f:
        opened = snapshot(os.fstat(f.fileno()))
        if opened != before: raise ValueError('member changed before readback')
        for chunk in iter(lambda: f.read(4*2**20), b''): h.update(chunk); count += len(chunk)
        if snapshot(os.fstat(f.fileno())) != opened: raise ValueError('member changed in readback')
    if regular(path) != before or count != size or h.hexdigest() != expected: raise ValueError('existing member SHA conflict')
    return dict(path=str(path), bytes=size, sha256=expected)

def validate_batch(q):
    worker = q['worker']; tpath, mpath = metadata(worker); qpath = queue_path(worker)
    if any(not isinstance(q.get(key),str) or not SHA.fullmatch(q[key]) for key in ('ticket_sha256','assignment_sha256','queue_sha256')): raise ValueError('complete original ticket/assignment/queue SHA required')
    traw, mraw, qraw = raw_read(tpath, 2**20), raw_read(mpath,64*2**20), raw_read(qpath,2**20)
    validate_queue(worker, qraw, q.get('ticket_sha256'), q.get('queue_sha256'))
    ticket, rows = validate(worker,traw,mraw,q.get('ticket_sha256'),q.get('assignment_sha256'))
    plan = q['plan']
    if len(encoded(plan)) > BATCH_METADATA_CAP or set(plan) != {'schema','target','manifest','commit','commit_sha256'} or plan['schema'] != 'ms_batchrpc_publication_plan_v1': raise ValueError('bounded plan schema')
    target = canonical(plan['target']); output = canonical(ticket['output_root']); anchor(output)
    try: parts = target.relative_to(output).parts
    except ValueError: raise ValueError('batch outside ticket output')
    c = plan['commit']; gpu = c.get('gpu'); run_id = c.get('run_id')
    if (type(gpu) is not int or gpu not in range(4) or not isinstance(run_id,str) or not re.fullmatch('g'+str(gpu)+r'-[0-9]+-[0-9a-f]{8}',run_id)
        or len(parts) != 3 or parts[:2] != ('batches','gpu'+str(gpu)) or not re.fullmatch(re.escape(run_id)+r'-batch[0-9]{3,}',parts[2])): raise ValueError('GPU/run/output binding')
    signatures = {Path(ref['path']).name:ref['sha256'] for ref in ticket['code_refs']}
    if (c.get('schema') != 'ms_unified_novlm_batch_commit_v1' or c.get('edition') != P.name or c.get('worker') != worker
        or c.get('assignment_sha256') != q['assignment_sha256'] or c.get('code_signatures') != signatures
        or c.get('quality_approved') is not False or c.get('vlm_called') is not False
        or hashlib.sha256(encoded(c)).hexdigest() != plan['commit_sha256']): raise ValueError('commit identity/SHA/code/policy')
    receipts = c.get('receipts'); indices = c.get('assignment_indices')
    if not isinstance(receipts,list) or not 1 <= len(receipts) <= BATCH_MAX_SOURCES or indices != [r.get('assignment_index') for r in receipts]: raise ValueError('bounded production source sequence')
    if any(type(i) is not int or not 0 <= i < len(rows) or i%4 != gpu for i in indices) or any(b-a != 4 for a,b in zip(indices,indices[1:])): raise ValueError('original lane source sequence')
    required = set(); archive_members = {}; diagnostic_groups = {}
    for index,r in zip(indices,receipts):
        row = rows[index]
        if r.get('source_row') != row: raise ValueError('immutable source row binding')
        status = r.get('status')
        if status == 'existing_claim_not_reassigned': continue
        if r.get('vlm_called') is not False: raise ValueError('receipt must declare VLM not called')
        if status == 'failed_sealed_no_retry': names = {'receipt.json'}
        elif status == 'extracted_rule_draft_unapproved' and r.get('draft_status') in ('draft_generated','partial','needs_regeneration'): names = FILES
        else: raise ValueError('source status binding')
        for name in names:
            rel = 'cases/'+row['split']+'/'+row['id']+'/'+name; required.add(str(target/rel)); archive_members[rel] = None
        if r.get('diagnostic_publications'):
            if status != 'failed_sealed_no_retry': raise ValueError('diagnostics cannot approve source')
            expected = {str(target/'diagnostics'/row['split']/row['id']/name) for name in DIAGNOSTIC_FILES}
            refs = r['diagnostic_publications']
            if not isinstance(refs,list) or len(refs) != 2 or {ref.get('path') for ref in refs} != expected: raise ValueError('independent diagnostic member set')
            required |= expected
            for ref in refs: diagnostic_groups[ref['path']] = ref
    required.add(str(target/'batch.tar'))
    manifest = plan['manifest']
    if not isinstance(manifest,list) or len(manifest) > 141 or [m.get('path') for m in manifest] != sorted(required): raise ValueError('exact sorted member manifest')
    total = 0; publications = []
    for m in manifest:
        if set(m) != {'path','bytes','sha256'} or type(m['bytes']) is not int or not 0 < m['bytes'] <= LIMIT or not isinstance(m['sha256'],str) or not SHA.fullmatch(m['sha256']): raise ValueError('member size/SHA/schema')
        p = canonical(m['path'])
        # Exact required paths above are constructed only from validated assigned rows.
        if not p.is_relative_to(target): raise ValueError('member outside original whitelist')
        if p.name == 'beatthis_rejected.npz' and m['bytes'] > 2*2**20 or p.name == 'beatthis_rejected.json' and m['bytes'] > 64*2**10: raise ValueError('diagnostic size cap')
        if m['path'] in diagnostic_groups and diagnostic_groups[m['path']] != m: raise ValueError('diagnostic receipt binding')
        total += m['bytes']; publications.append(dict(path=m['path'],sha256=m['sha256'],receipt=m))
        rel = p.relative_to(target).as_posix()
        if rel in archive_members: archive_members[rel] = m
    if total > LIMIT or c.get('publications') != publications: raise ValueError('aggregate capacity or full publication receipt binding')
    archive = next(m for m in manifest if m['path'] == str(target/'batch.tar'))
    if c.get('batch_sha256') != archive['sha256'] or c.get('unique_sources_attempted') != sum(r['status'] != 'existing_claim_not_reassigned' for r in receipts): raise ValueError('archive SHA/source count binding')
    return target,manifest,c,archive_members

def verify_archive(path, members):
    seen = set(); end = 0
    with tarfile.open(path,'r:') as pack:
        for m in pack:
            rel = m.name
            if not m.isfile() or m.issparse() or m.issym() or m.islnk() or rel in seen or rel not in members or str(Path(rel)) != rel or Path(rel).is_absolute() or '..' in Path(rel).parts: raise ValueError('archive member path/type/duplicate/set')
            spec = members[rel]
            if m.size != spec['bytes']: raise ValueError('archive member size')
            h = hashlib.sha256(); count = 0
            with pack.extractfile(m) as f:
                for chunk in iter(lambda:f.read(4*2**20),b''): h.update(chunk); count += len(chunk)
            if count != m.size or h.hexdigest() != spec['sha256']: raise ValueError('archive member SHA')
            seen.add(rel); end = m.offset_data + ((m.size+511)//512)*512
    if seen != set(members): raise ValueError('archive member set incomplete')
    with path.open('rb') as f:
        f.seek(end)
        for chunk in iter(lambda:f.read(4*2**20),b''):
            if any(chunk): raise ValueError('archive trailing payload')


def diagnostic_failures(beats, downbeats, duration):
    import numpy as np
    failures = []
    for name, values in (('beats', beats), ('downbeats', downbeats)):
        if values.ndim != 1: failures.append(name + '.ndim_eq_1')
        if not np.isfinite(values).all(): failures.append(name + '.all_finite')
        if np.any(values < 0): failures.append(name + '.nonnegative')
        if np.any(values > duration + .02): failures.append(name + '.within_audio_duration_plus_0.02')
        if values.ndim and np.any(np.diff(values) <= 0): failures.append(name + '.strictly_increasing')
    if beats.ndim == downbeats.ndim == 1 and any(not np.isclose(beat, beats, atol=1e-6, rtol=0).any() for beat in downbeats):
        failures.append('downbeats.belongs_to_beats_atol_1e-6_rtol_0')
    return failures

def verify_case_receipts(target,c):
    for r in c['receipts']:
        if r['status']=='existing_claim_not_reassigned': continue
        row=r['source_row']; path=target/'cases'/row['split']/row['id']/'receipt.json'
        if strict(raw_read(path,64*2**20)) != r: raise ValueError('case receipt contents differ from commit/source')
        if r.get('diagnostic_publications'):
            directory=target/'diagnostics'/row['split']/row['id']
            evidence=strict(raw_read(directory/'beatthis_rejected.json',64*2**10))
            array=directory/'beatthis_rejected.npz'; ref=next(ref for ref in r['diagnostic_publications'] if ref['path']==str(array))
            if (evidence.get('schema')!='ms_beatthis_rejection_diagnostic_v2' or evidence.get('source_row')!=row
                or type(evidence.get('assignment_index')) is not int or evidence['assignment_index']!=r['assignment_index']
                or not isinstance(r.get('actual_audio_sha256'),str) or not SHA.fullmatch(r['actual_audio_sha256'])
                or evidence.get('audio_sha256')!=r['actual_audio_sha256']
                or evidence.get('actual_audio_metadata')!=r.get('actual_audio_metadata')
                or not isinstance(evidence.get('actual_audio_metadata'),dict)
                or evidence['actual_audio_metadata'].get('duration_seconds')!=evidence.get('actual_audio_duration_seconds')
                or evidence.get('method')!='beat_this.inference.File2Beats'
                or evidence.get('original_rejection_policy_unchanged') is not True
                or evidence.get('original_rejection_policy')!='core303-304: ndim=1; finite; nonnegative; <=duration+0.02; strictly increasing; downbeats belong to beats atol=1e-6 rtol=0'
                or evidence.get('array_coverage')!='complete_bounded_numeric_candidate'
                or evidence.get('algorithm_sha256')!='85ea6781a433a7f3cfe50fa074f56e7d7c291a815dfbf7b0220d4eb9caf8dda9'
                or evidence.get('algorithm_sha256')!=c['code_signatures'].get('ms_unified_novlm_v1.py')
                or not isinstance(evidence.get('audit_sha256'),str) or not SHA.fullmatch(evidence['audit_sha256'])
                or evidence['audit_sha256']!=c.get('next_version_code_signatures',{}).get('beatthis_rejection_next_v2.py')
                or evidence.get('failed_predicates')!=r.get('failed_predicates') or not evidence.get('failed_predicates')
                or evidence.get('evidence_only') is not True or evidence.get('quality_approved') is not False
                or evidence.get('vlm_called') is not False or evidence.get('device')!='cuda:0'
                or evidence.get('float16') is not False or evidence.get('dbn') is not False
                or evidence.get('model_sha256')!='8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331'
                or not isinstance(evidence.get('algorithm_sha256'),str) or not SHA.fullmatch(evidence['algorithm_sha256'])
                or type(evidence.get('actual_audio_duration_seconds')) not in (int,float) or evidence['actual_audio_duration_seconds']<=0): raise ValueError('independent rejection diagnostic/source/model/config binding')
            raw=evidence.get('raw_arrays',{})
            if raw.get('path')!='beatthis_rejected.npz' or raw.get('bytes')!=ref['bytes'] or raw.get('sha256')!=ref['sha256']: raise ValueError('rejected arrays SHA/size binding')
            import numpy as np, zipfile, math
            with zipfile.ZipFile(array) as archive:
                items=archive.infolist()
                if len(items)!=2 or {item.filename for item in items}!={'beats.npy','downbeats.npy'} or sum(item.file_size for item in items)>2*2**20: raise ValueError('bounded diagnostic NPZ decompression')
                for item in items:
                    with archive.open(item) as f:
                        version=np.lib.format.read_magic(f)
                        if version!=(1,0): raise ValueError('unsupported diagnostic NPY version')
                        shape,fortran,dtype=np.lib.format.read_array_header_1_0(f,max_header_size=65536)
                        size=math.prod(shape)*dtype.itemsize
                        if dtype.hasobject or size>2*2**20 or size+f.tell()!=item.file_size: raise ValueError('bounded diagnostic NPY allocation')
            with np.load(array,allow_pickle=False) as arrays:
                if set(arrays.files)!={'beats','downbeats'}: raise ValueError('diagnostic NPZ member set')
                originals={}
                for name in ('beats','downbeats'):
                    actual=arrays[name]; expected=raw.get(name,{})
                    originals[name]=actual
                    if actual.dtype.hasobject or actual.dtype.kind not in 'biufc' or actual.nbytes>2*2**20 or expected!={'dtype':actual.dtype.str,'shape':list(actual.shape)}: raise ValueError('raw diagnostic dtype/shape/object contract')
                failures=diagnostic_failures(np.asarray(originals['beats'],dtype=float),np.asarray(originals['downbeats'],dtype=float),evidence['actual_audio_duration_seconds'])
                if not failures or failures!=evidence['failed_predicates']: raise ValueError('diagnostic predicates disagree with original bounded numeric arrays')

def inspect_batch(q, complete=False, require_commit=False):
    target,manifest,c,members = validate_batch(q); commit_path = target/'commit.json'
    anchor(commit_path)
    committed = os.path.lexists(commit_path)
    # Unknown status always inspects the existing commit before any publication.
    if committed: hash_regular(commit_path,len(encoded(c)),q['plan']['commit_sha256'])
    existing = []
    for m in manifest:
        p = canonical(m['path']); anchor(p)
        if os.path.lexists(p): hash_regular(p,m['bytes'],m['sha256']); existing.append(m['path'])
        elif committed or complete: raise ValueError('committed/required member missing')
    if committed or complete:
        verify_archive(target/'batch.tar',members); verify_case_receipts(target,c)
    if require_commit and not committed: raise ValueError('commit absent after publication')
    return dict(state='committed_verified' if committed else ('members_partial' if existing else 'absent'),
        existing=existing,commit_path=str(commit_path),commit_sha256=q['plan']['commit_sha256'],
        publications=c['publications'] if committed else [])

def stream_member(reader,spec,scope):
    p = canonical(spec['path']); mkdir(p.parent,scope); anchor(p); temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=p.parent,prefix='.batchrpc-',delete=False) as f:
            temporary = Path(f.name); h = hashlib.sha256(); count = 0
            while count < spec['bytes']:
                chunk = reader.read(min(4*2**20,spec['bytes']-count))
                if not chunk: raise ValueError('truncated bounded batch stream')
                f.write(chunk); h.update(chunk); count += len(chunk)
            if h.hexdigest() != spec['sha256']: raise ValueError('batch stream SHA/sequence mismatch')
            f.flush(); os.fsync(f.fileno())
        try: os.link(temporary,p,follow_symlinks=False)
        except FileExistsError: pass
        sync(p.parent); hash_regular(p,spec['bytes'],spec['sha256'])
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)

def dispatch(q, reader, writer):
    if q.get('op') not in ('publish_batch','inspect_batch','verify_batch','inspect_members'): return _old_dispatch(q,reader,writer)
    if canonical(q['path']) != canonical(q['plan']['target']): raise ValueError('RPC target binding')
    if q['op'] != 'publish_batch': return inspect_batch(q,q['op'] in ('verify_batch','inspect_members'),q['op']=='verify_batch')
    target,manifest,c,members = validate_batch(q); state = inspect_batch(q)
    for m in manifest: stream_member(reader,m,Path(P/'results'/q['worker']))
    if reader.read(1): raise ValueError('trailing batch stream bytes')
    for m in manifest: hash_regular(Path(m['path']),m['bytes'],m['sha256'])
    verify_archive(target/'batch.tar',members); verify_case_receipts(target,c)
    commit_raw = encoded(c)
    import io
    stream_member(io.BytesIO(commit_raw),dict(path=str(target/'commit.json'),bytes=len(commit_raw),sha256=q['plan']['commit_sha256']),Path(P/'results'/q['worker']))
    return inspect_batch(q,True,True)
'''
exec(COMMON_CODE, globals())
RELAY_CODE = COMMON_CODE + r'''
import base64, sys
try:
    if sys.argv[1] == '--request-stdin':
        prefix = sys.stdin.buffer.read(4)
        if len(prefix) != 4: raise ValueError('truncated request length')
        size = int.from_bytes(prefix, 'big')
        if not 0 < size <= BATCH_METADATA_CAP: raise ValueError('bounded request header size')
        raw = sys.stdin.buffer.read(size)
        if len(raw) != size: raise ValueError('truncated bounded request header')
        q = strict(raw)
    else: q = strict(base64.b64decode(sys.argv[1], validate=True))
    result = dispatch(q, sys.stdin.buffer, sys.stdout.buffer)
    if result is not None: sys.stdout.buffer.write(result if isinstance(result, bytes) else encoded(result))
    sys.stdout.buffer.flush()
except Exception as error:
    print("relay rejected: " + type(error).__name__ + ": " + str(error), file=sys.stderr, flush=True)
    sys.exit(1)
'''
RELAY_HOST = "11.239.36.34"
RPC_TIMEOUT = 600
ERROR_CAP = 8192

def request_header(q):
    raw = encoded(q)
    if not 0 < len(raw) <= BATCH_METADATA_CAP: raise ValueError('bounded request header size')
    return len(raw).to_bytes(4, 'big') + raw

def bounded_exchange(command, header, stream, timeout=RPC_TIMEOUT):
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    deadline = time.monotonic() + timeout
    selector = selectors.DefaultSelector(); pending = memoryview(header); output = bytearray(); errors = bytearray(); broken_input = False
    try:
        for pipe, events, name in ((process.stdin, selectors.EVENT_WRITE, 'stdin'), (process.stdout, selectors.EVENT_READ, 'stdout'), (process.stderr, selectors.EVENT_READ, 'stderr')):
            os.set_blocking(pipe.fileno(), False); selector.register(pipe, events, name)
        while selector.get_map() or process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0: raise TimeoutError('batch RPC whole exchange deadline exceeded')
            for key, _ in selector.select(min(remaining, .1)):
                pipe = key.fileobj
                if key.data == 'stdin':
                    if not pending:
                        chunk = stream.read(4*2**20)
                        if len(chunk) > 4*2**20: raise ValueError('unbounded local batch source')
                        pending = memoryview(chunk)
                    if not pending:
                        selector.unregister(pipe); pipe.close(); continue
                    try: count = os.write(pipe.fileno(), pending[:65536])
                    except BlockingIOError: continue
                    except BrokenPipeError:
                        broken_input = True
                        selector.unregister(pipe); pipe.close(); continue
                    pending = pending[count:]
                else:
                    try: chunk = os.read(pipe.fileno(), 65536)
                    except BlockingIOError: continue
                    if not chunk:
                        selector.unregister(pipe); pipe.close(); continue
                    buffer, cap = (output, BATCH_METADATA_CAP) if key.data == 'stdout' else (errors, ERROR_CAP)
                    if len(buffer) + len(chunk) > cap: raise ValueError('batch '+key.data+' exceeds bounded response cap')
                    buffer.extend(chunk)
        code = process.wait(timeout=max(.001, deadline-time.monotonic()))
        if code: raise RuntimeError(bytes(errors).decode('utf-8','replace') or 'batch acknowledgement unknown')
        if broken_input: raise RuntimeError('batch input closed before complete send; acknowledgement unknown')
        return strict(bytes(output))
    finally:
        selector.close()
        if process.poll() is None:
            process.terminate()
            try: process.wait(timeout=1)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=1)
        for pipe in (process.stdin, process.stdout, process.stderr):
            if pipe and not pipe.closed: pipe.close()

class BaseAccess:
    def __init__(self, worker, relay_host=None):
        self.worker, self.relay = worker, relay_host
        if worker not in WORKERS or relay_host not in (None, RELAY_HOST): raise ValueError("worker/relay not allowed")
        self.ticket_path, self.assignment_path = metadata(worker)
        self.queue_path = queue_path(worker)
        self.raw_evidence, self.copy_evidence, self.binding = {}, {}, {}
        self.rows, self.ticket, self.queue = [], None, None
    def request(self, op, path, **fields):
        q = dict(worker=self.worker, op=op, path=str(path), **self.binding, **fields)
        argument = "--request-stdin" if "plan" in fields else base64.b64encode(encoded(q)).decode()
        if "plan" in fields: request_header(q)
        command = "python3 -c " + shlex.quote(RELAY_CODE) + " " + shlex.quote(argument)
        return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=/tmp/ms_pilot_relay_known_hosts", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3", self.relay, command]
    def rpc(self, op, path, raw=False, data=None, stream=None, **fields):
        if self.relay is None:
            result = dispatch(dict(worker=self.worker, op=op, path=str(path), **self.binding, **fields), stream if stream is not None else io.BytesIO(data or b""), io.BytesIO())
            return result
        if 'plan' in fields:
            if data is not None or stream is not None: raise ValueError('metadata plan RPC cannot append payload')
            q = dict(worker=self.worker, op=op, path=str(path), **self.binding, **fields)
            return bounded_exchange(self.request(op, path, **fields), request_header(q), io.BytesIO(), RPC_TIMEOUT)
        r = subprocess.run(self.request(op, path, **fields), input=data, stdin=stream if stream is not None else (subprocess.DEVNULL if data is None else None), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600, check=False)
        if r.returncode: raise RuntimeError(r.stderr.decode("utf-8", "replace")[:8192] or "sealed relay failure; no retry")
        return r.stdout if raw else strict(r.stdout)
    def read(self, path, expected_sha=None):
        raw = self.rpc("read", path, raw=True, **({"expected_sha256": expected_sha} if expected_sha is not None else {}))
        self.raw_evidence[str(path)] = raw
        return raw
    def bind(self, ticket, rows, ticket_sha, assignment_sha):
        traw, mraw = self.read(self.ticket_path), self.read(self.assignment_path)
        qraw = self.read(self.queue_path); queue_sha = hashlib.sha256(qraw).hexdigest()
        queue = validate_queue(self.worker, qraw, ticket_sha, queue_sha)
        actual, actual_rows = validate(self.worker, traw, mraw, ticket_sha, assignment_sha)
        if actual != ticket or actual_rows != rows: raise ValueError("bind objects differ from original owned bytes")
        self.ticket, self.rows, self.queue = actual, actual_rows, queue
        self.binding = dict(ticket_sha256=ticket_sha, assignment_sha256=assignment_sha, queue_sha256=queue_sha)
        self.media = {row[k] for row in rows for k in ("video_path", "audio_path", "meta_path")}
    def stat(self, path): return self.rpc("stat", path)
    def exists(self, path): return self.rpc("exists", path)["exists"]
    def claim(self, path, record): return self.rpc("claim", path, record=record)["created"]
    def describe(self, path):
        return dict(original_path=str(path), accessible_path="ssh://" + self.relay + str(path) if self.relay else str(path), access_transport="ssh_stdout" if self.relay else "filesystem", accessible_node=self.relay or "worker_local")
    def copy(self, row, local_root, max_bytes, profile=None):
        started = time.perf_counter()
        if row not in self.rows: raise ValueError("copy outside own assignment")
        if type(max_bytes) is not int or not 0 <= max_bytes <= LIMIT: raise ValueError("copy max_bytes")
        destination = canonical(str(Path(local_root) / row["split"] / row["id"] / "inputs"))
        stamps = {k: self.stat(row[k]) for k in ("video_path", "audio_path")}
        if sum(s["bytes"] for s in stamps.values()) > max_bytes: raise ValueError("source exceeds copy byte budget")
        anchor(destination); destination.mkdir(parents=True); anchor(destination)
        copied, hashes, evidence = {}, {}, {}
        try:
            for key, name in (("video_path", "video.mp4"), ("audio_path", "audio.wav")):
                original, expected, target = row[key], stamps[key], destination / name
                process = None; source = None
                try:
                    if self.relay:
                        process = subprocess.Popen(self.request("copy", original, snapshot=expected, max_bytes=max_bytes), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
                        source = process.stdout
                        if strict(source.readline(65537)).get("source_stat_before") != expected: raise ValueError("source header mismatch")
                    else:
                        source = anchor(Path(original)).open("rb")
                        if snapshot(os.fstat(source.fileno())) != expected: raise ValueError("source open changed")
                    h, count = hashlib.sha256(), 0
                    with target.open("xb") as output:
                        while count < expected["bytes"]:
                            chunk = source.read(min(4*1024*1024, expected["bytes"]-count))
                            if not chunk: raise ValueError("short copy; partial retained")
                            output.write(chunk); h.update(chunk); count += len(chunk)
                        output.flush(); os.fsync(output.fileno())
                    if self.relay:
                        trailer = strict(source.readline(65537))
                        if trailer.get("source_stat_after") != expected or trailer.get("copied") != count or source.read(1): raise ValueError("source trailer mismatch")
                        stderr = process.stderr.read(8193)
                        if process.wait(timeout=600): raise RuntimeError(stderr.decode("utf-8", "replace") or "sealed stream failure")
                    elif source.read(1) or snapshot(os.fstat(source.fileno())) != expected or self.stat(original) != expected: raise ValueError("source changed")
                    copied[key], hashes[key] = target, h.hexdigest()
                    evidence[key] = dict(**self.describe(original), source_stat_before=expected, source_stat_after=expected, bytes=count, sha256=hashes[key])
                finally:
                    if source is not None: source.close()
                    if process is not None:
                        if process.poll() is None:
                            process.terminate()
                            try: process.wait(timeout=10)
                            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
                        process.stderr.close()
            self.copy_evidence[(row["split"], row["id"])] = evidence
            return copied, hashes
        finally:
            if profile is not None: profile["copy"] = time.perf_counter()-started
    def publish_file(self, local, target, expected_sha):
        local = Path(local); before = regular(local)
        with local.open("rb") as stream: result = self.rpc("publish", target, stream=stream, size=before["bytes"], sha256=expected_sha)
        if regular(local) != before or result != dict(path=str(target), bytes=before["bytes"], sha256=expected_sha): raise ValueError("publication receipt/source mismatch")
        return result
    def publish_json(self, value, target):
        raw = encoded(value)
        return self.rpc("publish", target, data=raw, size=len(raw), sha256=hashlib.sha256(raw).hexdigest())

class Access(BaseAccess):
    def inspect_plan(self,plan): return self.rpc('inspect_batch',plan['target'],plan=plan)
    def inspect_members(self,plan): return self.rpc('inspect_members',plan['target'],plan=plan)
    def verify_plan(self,plan):
        result=self.rpc('verify_batch',plan['target'],plan=plan)
        if result.get('state')!='committed_verified' or result.get('commit_sha256')!=plan['commit_sha256'] or result.get('publications')!=plan['commit']['publications']: raise ValueError('batch complete readback receipt mismatch')
        # The exact commit is also returned through the pre-existing expected-SHA read API.
        if self.read(str(Path(plan['target'])/'commit.json'),expected_sha=plan['commit_sha256'])!=encoded(plan['commit']): raise ValueError('complete commit expectedSHA readback mismatch')
        return result
    def batch_exchange(self,plan,stream):
        if self.relay is None:
            return dispatch(dict(worker=self.worker,op='publish_batch',path=plan['target'],**self.binding,plan=plan),stream,io.BytesIO())
        q=dict(worker=self.worker,op='publish_batch',path=plan['target'],**self.binding,plan=plan)
        return bounded_exchange(self.request('publish_batch',plan['target'],plan=plan),request_header(q),stream,RPC_TIMEOUT)
    def publish_prepared(self,saved):
        from batchrpc_publication_next_v2 import publish
        return publish(self,saved)
    def recover_prepared(self,saved):
        from batchrpc_publication_next_v2 import recover
        return recover(self,saved)
