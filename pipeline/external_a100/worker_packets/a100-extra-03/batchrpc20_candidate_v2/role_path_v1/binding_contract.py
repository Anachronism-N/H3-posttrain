"""Read-only original ticket/assignment/queue validation, frozen derivative."""
import hashlib, json, re
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
def metadata(worker):
    if worker not in WORKERS: raise ValueError("worker not allowed")
    anchor(P)
    t, m = P / "assignments" / (worker + ".ticket.json"), P / "assignments" / (worker + ".jsonl")
    return t, m
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

