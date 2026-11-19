"""CPU-only fixed-fixture publication candidate. No deployment, media, GPU or claims."""
import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import sys
import tarfile
import time
import uuid
sys.dont_write_bytecode = True
from binding_contract import P, PREVIOUS_P, SAFE, SHA, FILES, strict, encoded, validate, validate_queue

OWN = Path("/tmp/ms_pose2d_draft_20261026T1427Z/launcher/batchrpc20_candidate_v2")
CENTRAL = P / "results/a100-extra-04/publication_experiments"
WORKER = "a100-extra-04"
RELAY = "11.239.36.34"
HEADER_CAP, COUNT_CAP, BLOCK = 2**20, 101, 2**20
TOTAL_CAP, OUTPUT_CAP, CACHE_CAP = 8*2**30, 4*2**30, 8*2**20
TICKET_SHA = "394d4a526b2912c7ed96431ca54146e5756f3e9a2e62ee908db6141a4378a6c3"
ASSIGNMENT_SHA = "17bd9595494df97cfc9a36814a611f92cb12fa8ad2af387d4b068dae9fcffced"
QUEUE_SHA = "d1162cf42d77ab34b876e34322017e04ab6651a69dcf83831dc4fe428981de35"
FIXTURES = {
    "mixed": ("gpu2/g2-1793048206163767809-b2d4af40-batch058", "e1cc37df6d0edaf4645b2745c121ce0ef1bb56499949212f97ed15426799de58", 2),
    "all_success": ("gpu0/g0-1793047974640888520-c1389581-batch058", "e3d09c01eea4b004fc9ea0f90d1e506f55565da2b66fdcab539ef736923e0ea7", 0),
}
ARMS = {"baseline", "batch", "cpu_mixed", "cpu_all_success", "fault_short", "fault_bad", "fault_sha", "fault_extra", "fault_symlink", "already_good"}
OPS = {"prepare", "batch", "file", "fallback_file", "finalize", "readback", "fault_readback"}
IO_DEADLINE = float("inf")
RPC_PROCESSES_STARTED = 0

def check_deadline():
    if time.monotonic() >= IO_DEADLINE:
        raise TimeoutError("finite test publication deadline; no signals, no retry")

def timeout_check(q):
    duration=q.get("timeout_seconds")
    if type(duration) not in (int,float) or not 0 < duration <= 600:
        raise ValueError("finite RPC duration <=600seconds")

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def run_root(run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{0,63}", run_id):
        raise ValueError("fixed test run id")
    return CENTRAL / ("batchrpc20_candidate_v2_" + run_id)

def target_root(run_id, arm, fixture):
    if arm not in ARMS or fixture not in FIXTURES:
        raise ValueError("fixed test arm/fixture")
    return run_root(run_id) / arm / fixture

def canonical(path):
    text = str(path)
    if not Path(text).is_absolute() or str(Path(text)) != text or ".." in Path(text).parts or "\x00" in text:
        raise ValueError("canonical absolute path required")
    return Path(text)

def directory_fd(path, create=False, scope=None):
    path = canonical(path)
    if create and (scope is None or (path != scope and scope not in path.parents)):
        raise ValueError("mkdir outside own test scope")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    current = Path("/")
    try:
        for part in path.parts[1:]:
            check_deadline()
            current /= part
            try:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            except FileNotFoundError:
                if not create or (current != scope and scope not in current.parents):
                    raise
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                    os.fsync(fd)
                except FileExistsError:
                    pass
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise

def open_regular(path):
    path = canonical(path)
    parent = directory_fd(path.parent)
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    finally:
        os.close(parent)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("opened FD not regular")
    return os.fdopen(fd, "rb")

def bounded_read(path, cap, expected=None):
    check_deadline()
    with open_regular(path) as stream:
        before = os.fstat(stream.fileno())
        if before.st_size > cap:
            raise ValueError("metadata cap")
        raw = stream.read(cap + 1)
        after = os.fstat(stream.fileno())
        if len(raw) != before.st_size or before.st_size != after.st_size or len(raw) > cap:
            raise ValueError("metadata short/changing FD")
        if expected is not None and sha(raw) != expected:
            raise ValueError("metadata complete expected SHA")
        return raw

def boundary_check(item):
    if not isinstance(item, dict) or set(item) != {"name", "bytes", "sha256"}:
        raise ValueError("explicit file boundary schema")
    name = item["name"]
    if name != "batch.tar":
        if not isinstance(name, str):
            raise ValueError("file member name")
        parts = name.split("/")
        if len(parts) != 4 or parts[0] != "cases" or not all(SAFE.fullmatch(x) for x in parts[1:3]) or parts[3] not in FILES:
            raise ValueError("unsafe/noncanonical case member")
    if type(item["bytes"]) is not int or not 0 < item["bytes"] <= OUTPUT_CAP or not isinstance(item["sha256"], str) or not SHA.fullmatch(item["sha256"]):
        raise ValueError("output cap or SHA")

def boundaries_check(items):
    if not isinstance(items, list) or not 1 <= len(items) <= COUNT_CAP:
        raise ValueError("file boundary count")
    seen, total = set(), 0
    for item in items:
        boundary_check(item)
        if item["name"] in seen:
            raise ValueError("duplicate file boundary")
        seen.add(item["name"]); total += item["bytes"]
    if total + HEADER_CAP > TOTAL_CAP:
        raise ValueError("total streaming cap8GiB including commit/header reservation")
    return items

def fixture_proof(fixture):
    if fixture not in FIXTURES:
        raise ValueError("fixed fixture only")
    tpath = P / "assignments" / (WORKER + ".ticket.json")
    mpath = P / "assignments" / (WORKER + ".jsonl")
    qpath = P / (WORKER + ".queue.json")
    # Exactly one complete SHA validation per metadata file per receiver request.
    traw = bounded_read(tpath, HEADER_CAP)
    mraw = bounded_read(mpath, 64*2**20)
    qraw = bounded_read(qpath, HEADER_CAP)
    _, rows = validate(WORKER, traw, mraw, TICKET_SHA, ASSIGNMENT_SHA)
    validate_queue(WORKER, qraw, TICKET_SHA, QUEUE_SHA)
    sub, expected, gpu = FIXTURES[fixture]
    root = P / "results" / WORKER / "batches" / sub
    raw = bounded_read(root / "commit.json", HEADER_CAP, expected)
    commit = strict(raw)
    if (commit.get("schema") != "ms_unified_novlm_batch_commit_v1" or commit.get("edition") != P.name
        or commit.get("worker") != WORKER or type(commit.get("gpu")) is not int or commit["gpu"] != gpu
        or commit.get("assignment_sha256") != ASSIGNMENT_SHA or len(commit.get("receipts", [])) != 20
        or not isinstance(commit.get("code_signatures"), dict) or not commit["code_signatures"]
        or any(not SHA.fullmatch(str(v)) for v in commit["code_signatures"].values())):
        raise ValueError("original batch identity/signature evidence")
    indices = commit.get("assignment_indices")
    if indices != [r.get("assignment_index") for r in commit["receipts"]] or len(set(indices)) != 20:
        raise ValueError("exact20 original indices")
    names = set()
    statuses = []
    for receipt in commit["receipts"]:
        index = receipt.get("assignment_index")
        if type(index) is not int or not 0 <= index < len(rows) or index % 4 != gpu or receipt.get("source_row") != rows[index]:
            raise ValueError("original source row/index%4/GPU")
        for key in ("worker", "worker_id"):
            if key in receipt and receipt[key] != WORKER:
                raise ValueError("original receipt worker")
        for key in ("gpu", "physical_gpu_index", "physical_gpu"):
            if key in receipt and (type(receipt[key]) is not int or receipt[key] != gpu):
                raise ValueError("original receipt GPU")
        status = receipt.get("status")
        statuses.append(status)
        if status not in ("extracted_rule_draft_unapproved", "failed_sealed_no_retry"):
            raise ValueError("unsupported sealed source status")
        row = rows[index]
        allowed = {"receipt.json"} if status == "failed_sealed_no_retry" else FILES
        names.update("cases/" + row["split"] + "/" + row["id"] + "/" + n for n in allowed)
    if fixture == "mixed" and statuses.count("failed_sealed_no_retry") != 1:
        raise ValueError("mixed fixture exact19partial+1sealedfailed")
    if fixture == "all_success" and statuses.count("extracted_rule_draft_unapproved") != 20:
        raise ValueError("all-success exact20 partial/unapproved, not semantic approval")
    boundaries, path_map = [], []
    for pub in commit.get("publications", []):
        if set(pub) != {"path", "sha256", "receipt"} or not isinstance(pub["receipt"], dict) or set(pub["receipt"]) != {"path", "bytes", "sha256"}:
            raise ValueError("production receipt shape")
        path = canonical(pub["path"])
        name = str(path.relative_to(root))
        rec = pub["receipt"]
        if rec["path"] != str(path) or rec["sha256"] != pub["sha256"]:
            raise ValueError("original receipt mismatch")
        boundaries.append(dict(name=name, bytes=rec["bytes"], sha256=pub["sha256"]))
        path_map.append(dict(original_path=str(path), name=name))
    boundaries_check(boundaries)
    byname = {b["name"]: b for b in boundaries}
    if set(byname) != names | {"batch.tar"} or byname["batch.tar"]["sha256"] != commit.get("batch_sha256"):
        raise ValueError("exact original source file set/archive SHA")
    boundaries = sorted((b for b in boundaries if b["name"] != "batch.tar"), key=lambda b:b["name"]) + [byname["batch.tar"]]
    return dict(fixture=fixture, original_commit=commit, original_commit_path=str(root / "commit.json"),
                original_commit_sha256=expected, boundaries=boundaries, original_paths={m["name"]:m["original_path"] for m in path_map},
                source_count=20, gpu=gpu, assignment_indices=indices, source_rows=[r["source_row"] for r in commit["receipts"]],
                ticket_sha256=TICKET_SHA, assignment_sha256=ASSIGNMENT_SHA, queue_sha256=QUEUE_SHA)

def request(proof, run_id, arm, op, names=None):
    items = proof["boundaries"] if names is None else [b for b in proof["boundaries"] if b["name"] in names]
    return dict(op=op, run_id=run_id, arm=arm, fixture=proof["fixture"], worker=WORKER, execution_node=WORKER,
                gpu=proof["gpu"], assignment_indices=proof["assignment_indices"], source_rows=proof["source_rows"],
                test_only=True, no_annotation=True, newsource=0, original_commit_sha256=proof["original_commit_sha256"],
                ticket_sha256=TICKET_SHA, assignment_sha256=ASSIGNMENT_SHA, queue_sha256=QUEUE_SHA, files=items, timeout_seconds=600)

def request_check(q, proof):
    if not isinstance(q, dict) or set(q) != {"op", "run_id", "arm", "fixture", "worker", "execution_node", "gpu", "assignment_indices", "source_rows", "test_only", "no_annotation", "newsource", "original_commit_sha256", "ticket_sha256", "assignment_sha256", "queue_sha256", "files", "timeout_seconds"}:
        raise ValueError("arbitrary target/production/extra options rejected")
    timeout_check(q)
    root = target_root(q["run_id"], q["arm"], q["fixture"])
    if q["op"] not in OPS or q["test_only"] is not True or q["no_annotation"] is not True or type(q["newsource"]) is not int or q["newsource"] != 0:
        raise ValueError("test-only publication gate")
    expected = request(proof, q["run_id"], q["arm"], q["op"])
    if any(q[k] != expected[k] or (k == "gpu" and type(q[k]) is not int) for k in expected if k not in ("files", "timeout_seconds")):
        raise ValueError("original worker/GPU/index/row/full SHA binding")
    boundaries_check(q["files"])
    if q["op"] in ("file", "fallback_file"):
        if len(q["files"]) != 1 or q["files"][0] not in proof["boundaries"]:
            raise ValueError("one exact original file fallback/baseline boundary")
    elif q["files"] != proof["boundaries"]:
        raise ValueError("exact full20source boundary list")
    if len(encoded(q)) > HEADER_CAP:
        raise ValueError("header cap1MiB")
    return root

def expected_commit(proof, root):
    commit = copy.deepcopy(proof["original_commit"])
    pubs, mapping = [], []
    for item in proof["boundaries"]:
        path = str(root / item["name"])
        receipt = dict(path=path, bytes=item["bytes"], sha256=item["sha256"])
        pubs.append(dict(path=path, sha256=item["sha256"], receipt=receipt))
        mapping.append(dict(original_path=proof["original_paths"][item["name"]], target_path=path, bytes=item["bytes"], sha256=item["sha256"]))
    commit["publications"] = pubs
    commit.update(test_only=True, no_annotation=True, newsource=0, source_count=20, model_calls=0, sourceclaims=0,
                  original_commit_path=proof["original_commit_path"], original_commit_sha256=proof["original_commit_sha256"],
                  original_to_test_paths=mapping, candidate="batchrpc20_candidate_v2", actual_fullpipeline_speedup=False)
    raw = encoded(commit)
    if len(raw) > HEADER_CAP:
        raise ValueError("commit metadata cap")
    return commit, raw

def verify_file(path, item):
    boundary_check(item) if item["name"] != "commit.json" else None
    with open_regular(path) as stream:
        before = os.fstat(stream.fileno())
        if before.st_size != item["bytes"] or before.st_size > OUTPUT_CAP:
            raise ValueError("sealed reject: opened target FD expected length")
        h, count = hashlib.sha256(), 0
        while count < item["bytes"]:
            check_deadline()
            chunk = stream.read(min(BLOCK, item["bytes"] - count))
            if not chunk:
                raise ValueError("sealed reject: readback short")
            h.update(chunk); count += len(chunk)
        after = os.fstat(stream.fileno())
        if stream.read(1) or after.st_size != before.st_size or h.hexdigest() != item["sha256"]:
            raise ValueError("sealed reject: opened target FD complete SHA")
    return dict(path=str(path), bytes=count, sha256=h.hexdigest())

def write_stream(path, reader, size, expected, scope):
    path = canonical(path)
    if path == scope or scope not in path.parents or path.name.startswith(".") or type(size) is not int or not 0 < size <= OUTPUT_CAP or not SHA.fullmatch(str(expected)):
        raise ValueError("test target/budget/SHA gate")
    fd = directory_fd(path.parent, create=True, scope=scope)
    temporary = ".batchrpc20-" + uuid.uuid4().hex
    opened = False
    try:
        tfd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
        opened = True
        h, copied = hashlib.sha256(), 0
        with os.fdopen(tfd, "wb") as out:
            while copied < size:
                check_deadline()
                chunk = reader.read(min(BLOCK, size-copied))
                if not chunk:
                    raise ValueError("short finite publication stream; commit absent")
                if len(chunk) > size-copied:
                    raise ValueError("reader violated file boundary")
                out.write(chunk); h.update(chunk); copied += len(chunk)
            if h.hexdigest() != expected:
                raise ValueError("publication file expected SHA; commit absent")
            out.flush(); os.fsync(out.fileno())
        check_deadline()
        try:
            os.link(temporary, path.name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
        except FileExistsError:
            pass
        os.fsync(fd)
        # Read back the exact original target FD, never the temp FD or incoming bytes.
        target_fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(target_fd, "rb") as original:
            before = os.fstat(original.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size != size:
                raise ValueError("sealed reject: existing target type/length; preserved")
            digest, n = hashlib.sha256(), 0
            while n < size:
                check_deadline()
                chunk = original.read(min(BLOCK, size-n))
                if not chunk:
                    raise ValueError("sealed reject: existing target short; preserved")
                digest.update(chunk); n += len(chunk)
            if original.read(1) or os.fstat(original.fileno()).st_size != size or digest.hexdigest() != expected:
                raise ValueError("sealed reject: existing target different SHA; preserved")
        return dict(path=str(path), bytes=size, sha256=expected)
    finally:
        if opened:
            os.unlink(temporary, dir_fd=fd)
        os.close(fd)

def intent(proof, root, require_existing=False):
    record = encoded(dict(test_only=True, original_commit_sha256=proof["original_commit_sha256"],
                          target=str(root), boundaries=proof["boundaries"], payload_binding=sha(encoded(proof["boundaries"]))))
    path = root / "publication.intent.json"
    if require_existing:
        if bounded_read(path, HEADER_CAP, sha(record)) != record:
            raise ValueError("fallback intent exact existing payload/target")
    else:
        write_stream(path, io.BytesIO(record), len(record), sha(record), run_root(root.parts[-3].removeprefix("batchrpc20_candidate_v2_")))

def complete(proof, root, scope):
    commit, raw = expected_commit(proof, root)
    for item, pub in zip(proof["boundaries"], commit["publications"]):
        if verify_file(root / item["name"], item) != pub["receipt"]:
            raise ValueError("production-shaped final file receipt")
    receipt = write_stream(root / "commit.json", io.BytesIO(raw), len(raw), sha(raw), scope)
    checked = bounded_read(root / "commit.json", HEADER_CAP, sha(raw))
    if checked != raw:
        raise ValueError("commit-last complete FD expected SHA/bytes")
    return dict(test_only=True, commit=receipt, publications=commit["publications"], model_calls=0, sourceclaims=0,
                all_file_tar_commit_readback=True, source_count=20)

def prepare_request(run_id, fixture):
    run_root(run_id)
    if fixture not in FIXTURES:
        raise ValueError("fixed prepare fixture")
    return dict(op="prepare", run_id=run_id, arm="batch", fixture=fixture, worker=WORKER, execution_node=WORKER,
                test_only=True, no_annotation=True, newsource=0, original_commit_sha256=FIXTURES[fixture][1],
                ticket_sha256=TICKET_SHA, assignment_sha256=ASSIGNMENT_SHA, queue_sha256=QUEUE_SHA, timeout_seconds=600)

def receive(q, reader, proof=None):
    # No bytes are published before full original metadata binding validation.
    if not isinstance(q, dict):
        raise ValueError("RPC JSON object only")
    proof = fixture_proof(q.get("fixture")) if proof is None else proof
    if q.get("op") == "prepare":
        expected = prepare_request(q.get("run_id"), q.get("fixture"))
        timeout_check(q)
        comparable=dict(q);comparable["timeout_seconds"]=600
        if comparable != expected or type(q.get("newsource")) is not int or q.get("test_only") is not True or q.get("no_annotation") is not True or reader.read(1):
            raise ValueError("fixed empty-payload prepare gate")
        return dict(proof=proof, model_calls=0, sourceclaims=0)
    root = request_check(q, proof)
    op, scope = q["op"], run_root(q["run_id"])
    if op in ("prepare", "readback", "finalize", "fault_readback") and reader.read(1):
        raise ValueError("unexpected bytes for metadata operation")
    if op == "prepare":
        return dict(proof=proof, model_calls=0, sourceclaims=0)
    if op == "fault_readback":
        if q["arm"] != "fault_short":
            raise ValueError("only fixed second-file fault branch")
        intent(proof,root,require_existing=True)
        first,second=proof["boundaries"][:2]
        checked=verify_file(root/first["name"],first)
        for absent in (root/second["name"],root/"commit.json"):
            fd=directory_fd(absent.parent)
            try:
                try:os.stat(absent.name,dir_fd=fd,follow_symlinks=False)
                except FileNotFoundError:pass
                else:raise ValueError("fault readback expected absent second file and commit")
            finally:os.close(fd)
        return dict(test_only=True,first_file_receipt=checked,second_file_absent=True,commit_absent=True,
                    same_target=str(root),original_commit_sha256=proof["original_commit_sha256"])
    if op == "readback":
        c, raw = expected_commit(proof, root)
        for item, pub in zip(proof["boundaries"], c["publications"]):
            if verify_file(root / item["name"], item) != pub["receipt"]:
                raise ValueError("readback production-shaped file receipt")
        if bounded_read(root / "commit.json", HEADER_CAP, sha(raw)) != raw:
            raise ValueError("commit exact full SHA")
        return dict(commit=dict(path=str(root/"commit.json"), bytes=len(raw), sha256=sha(raw)),
                    publications=c["publications"], all_file_tar_commit_readback=True, test_only=True, model_calls=0, sourceclaims=0)
    intent(proof, root, require_existing=op in ("fallback_file", "finalize"))
    if op == "finalize":
        return complete(proof, root, scope)
    receipts = []
    for item in q["files"]:
        receipt = write_stream(root / item["name"], reader, item["bytes"], item["sha256"], scope)
        receipts.append(dict(path=receipt["path"], sha256=receipt["sha256"], receipt=receipt))
    if reader.read(1):
        raise ValueError("extra bytes beyond final explicit file boundary; commit absent")
    if op == "batch":
        return complete(proof, root, scope)
    return dict(test_only=True, publications=receipts, model_calls=0, sourceclaims=0)

def extract_cached_archive(archive, proof, cache_root):
    items = {b["name"]:b for b in proof["boundaries"] if b["name"] != "batch.tar"}
    taritem = proof["boundaries"][-1]
    if taritem["bytes"] > CACHE_CAP or sum(b["bytes"] for b in items.values()) > CACHE_CAP:
        raise ValueError("fixture archive/expanded cache cap8MiB")
    verify_file(archive, taritem)
    found, total = set(), 0
    with open_regular(archive) as source, tarfile.open(fileobj=source, mode="r|*") as pack:
        for member in pack:
            if member.name not in items:
                raise ValueError("unexpected/unsafe tar member")
            boundary_check(items[member.name])
            if member.name in found or not member.isreg() or member.linkname or set(member.pax_headers)-{"mtime","atime","ctime"}:
                raise ValueError("tar duplicate/symlink/hardlink/directory/unsafePAX")
            if member.size != items[member.name]["bytes"]:
                raise ValueError("tar expected size")
            total += member.size
            if total > CACHE_CAP:
                raise ValueError("cache expansion cap8MiB")
            incoming = pack.extractfile(member)
            item = items[member.name]
            write_stream(cache_root / member.name, incoming, item["bytes"], item["sha256"], OWN)
            if incoming.read(1):
                raise ValueError("member extra bytes")
            found.add(member.name)
    if found != set(items):
        raise ValueError("archive incomplete exact expected members")
    return total

class CacheReader:
    def __init__(self, cache, items):
        self.cache, self.items, self.i, self.stream = cache, items, 0, None
    def read(self, n):
        if n <= 0 or n > BLOCK:
            raise ValueError("bounded stream reads only")
        while self.i < len(self.items):
            if self.stream is None:
                self.stream = open_regular(self.cache / self.items[self.i]["name"])
            raw = self.stream.read(n)
            if raw:
                return raw
            self.stream.close(); self.stream = None; self.i += 1
        return b""
    def close(self):
        if self.stream is not None:
            self.stream.close(); self.stream = None

def local_cache(run_id, fixture):
    run_root(run_id)
    if fixture not in FIXTURES:
        raise ValueError("cache fixed fixture")
    return OWN / "runs" / run_id / "cache" / fixture

def local_artifact(run_id, name, value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+\.json", name):
        raise ValueError("local own report only")
    run_root(run_id)
    raw = encoded(value)
    return write_stream(OWN / "runs" / run_id / name, io.BytesIO(raw), len(raw), sha(raw), OWN)

def prepare_local(run_id, fixture):
    proof = fixture_proof(fixture)
    cache = local_cache(run_id, fixture)
    item = proof["boundaries"][-1]
    if item["bytes"] > CACHE_CAP:
        raise ValueError("fixture archive cap8MiB")
    try:
        verify_file(cache / "batch.tar", item)
    except FileNotFoundError:
        with open_regular(proof["original_paths"]["batch.tar"]) as stream:
            write_stream(cache / "batch.tar", stream, item["bytes"], item["sha256"], OWN)
            if stream.read(1):
                raise ValueError("original archive extra bytes")
    # A valid retained cache is reused; no original archive/source rerun after interruption.
    expanded = extract_cached_archive(cache / "batch.tar", proof, cache)
    if expanded + item["bytes"] > CACHE_CAP:
        raise ValueError("archive+expanded fixture local cache cap8MiB")
    return proof, cache

def ssh_argv():
    # Fixed approved worker04 -> worker03 channel. No SSH config/key/known-host writes.
    candidate_source = bounded_read(OWN / "candidate.py", HEADER_CAP).decode()
    binding_source = bounded_read(OWN / "binding_contract.py", HEADER_CAP).decode()
    remote = "import sys,types\nsys.dont_write_bytecode=True\nm=types.ModuleType('binding_contract')\nexec(" + repr(binding_source) + ",m.__dict__)\nsys.modules['binding_contract']=m\n__name__='batchrpc20_receiver'\n" + candidate_source + "\nremote_main()\n"
    return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=/tmp/ms_pilot_relay_known_hosts",
            "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3", RELAY, "python3 -c " + shlex.quote(remote)]

def rpc(q, cache=None, deadline=None, prepare=False, inject_second_file_short=False):
    global RPC_PROCESSES_STARTED
    if len(encoded(q)) > HEADER_CAP:
        raise ValueError("outgoing header cap")
    if inject_second_file_short and (q.get("op")!="batch" or q.get("arm")!="fault_short" or len(q.get("files",[]))<2):
        raise ValueError("only fixed test fault_short second-file disconnect")
    remaining = deadline - time.monotonic()
    q=dict(q,timeout_seconds=min(600,remaining))
    if remaining <= 0:
        raise TimeoutError("finite experiment deadline; no retry")
    spool_root = OWN / "runs" / q["run_id"] / "rpc_spools"
    fd = directory_fd(spool_root, create=True, scope=OWN)
    token = uuid.uuid4().hex
    payload_path, out_path, err_path = (spool_root / (token+s) for s in (".in", ".out", ".err"))
    os.close(fd)
    # Disk spool is finite; no whole-batch memory and no detached or signalled subprocesses.
    with payload_path.open("xb") as incoming:
        incoming.write(encoded(q))
        if q["op"] in ("batch", "file", "fallback_file"):
            reader = CacheReader(cache, q["files"])
            try:
                cutoff=q["files"][0]["bytes"]+q["files"][1]["bytes"]//2 if inject_second_file_short else None
                written=0
                while cutoff is None or written<cutoff:
                    if time.monotonic()>=deadline:
                        raise TimeoutError("finite payload spool deadline; no RPC launched")
                    chunk = reader.read(BLOCK if cutoff is None else min(BLOCK,cutoff-written))
                    if not chunk:
                        break
                    incoming.write(chunk);written+=len(chunk)
            finally:
                reader.close()
        incoming.flush(); os.fsync(incoming.fileno())
    # subprocess.run timeout sends a signal, forbidden here: remote finite deadline plus finite SSH channel.
    env = dict(os.environ)
    env["BATCHRPC20_DEADLINE"] = str(deadline)
    with payload_path.open("rb") as incoming, out_path.open("xb") as outgoing, err_path.open("xb") as error:
        proc = subprocess.Popen(ssh_argv(), stdin=incoming, stdout=outgoing, stderr=error, env=env)
        RPC_PROCESSES_STARTED+=1
        while proc.poll() is None:
            if time.monotonic() >= deadline:
                # Do not terminate/kill/signal: retain handle/artifacts and stop the experiment.
                raise TimeoutError("deadline exceeded; process retained PID="+str(proc.pid)+" stdout="+str(out_path))
            time.sleep(0.05)
    err = bounded_read(err_path, HEADER_CAP)
    if proc.returncode:
        raise RuntimeError("sealed receiver reject; no AB retry; pid="+str(proc.pid)+" out="+str(out_path)+" err="+str(err_path)+" "+err.decode("utf-8", "replace")[:8192])
    if prepare:
        with open_regular(out_path) as source:
            header = source.readline(HEADER_CAP+1)
            if len(header)>HEADER_CAP or not header.endswith(b"\n"):
                raise ValueError("prepare response bounded header")
            result = strict(header); proof = result["proof"]
            comparable=dict(q);comparable["timeout_seconds"]=600
            if comparable != prepare_request(q["run_id"], proof["fixture"]) or proof["original_commit_sha256"] != FIXTURES[q["fixture"]][1] or sha(encoded(proof["original_commit"])) != proof["original_commit_sha256"]:
                raise ValueError("prepared exact original commit SHA and fixed identity")
            request_check(request(proof, q["run_id"], "batch", "batch"), proof)
            item = proof["boundaries"][-1]
            if item["bytes"] > CACHE_CAP:
                raise ValueError("prepare cache8MiB")
            write_stream(cache / "batch.tar", source, item["bytes"], item["sha256"], OWN)
            if source.read(1):
                raise ValueError("prepare stream extra")
        expanded = extract_cached_archive(cache / "batch.tar", proof, cache)
        if expanded+item["bytes"] > CACHE_CAP:
            raise ValueError("full cache cap8MiB")
        return result
    return strict(bounded_read(out_path, HEADER_CAP))

def remote_main():
    global IO_DEADLINE
    try:
        header = sys.stdin.buffer.readline(HEADER_CAP+1)
        if len(header)>HEADER_CAP or not header.endswith(b"\n"):
            raise ValueError("finite header <=1MiB")
        q = strict(header)
        timeout_check(q);IO_DEADLINE=time.monotonic()+q["timeout_seconds"]
        result = receive(q, sys.stdin.buffer)
        sys.stdout.buffer.write(encoded(result)); sys.stdout.buffer.flush()
        if q["op"] == "prepare":
            proof = result["proof"]; item = proof["boundaries"][-1]
            if item["bytes"] > CACHE_CAP:
                raise ValueError("fixture cache cap8MiB")
            # Original archive one open/read only; hash while streaming, never load it into RAM.
            with open_regular(proof["original_paths"]["batch.tar"]) as source:
                h, n = hashlib.sha256(), 0
                while n < item["bytes"]:
                    check_deadline()
                    chunk = source.read(min(BLOCK, item["bytes"]-n))
                    if not chunk:
                        raise ValueError("original archive short")
                    h.update(chunk); n += len(chunk); sys.stdout.buffer.write(chunk)
                if source.read(1) or h.hexdigest()!=item["sha256"]:
                    raise ValueError("original archive SHA/length")
            sys.stdout.buffer.flush()
    except Exception as error:
        print("sealed reject: "+type(error).__name__+": "+str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)

def topology_fault_fallback(proof,cache,run_id,deadline):
    start=time.monotonic();calls=0;record=dict(test_only=True,arm="fault_short",same_target=str(target_root(run_id,"fault_short",proof["fixture"])),
                                           source_generation_repeated=False,model_calls=0,sourceclaims=0,actual_rpc_calls=0)
    try:
        calls+=1
        try:
            rpc(request(proof,run_id,"fault_short","batch"),cache,deadline,inject_second_file_short=True)
        except RuntimeError as error:
            if "short finite publication stream; commit absent" not in str(error):
                raise
            record["injected_failure_retained"]=str(error)
        else:
            raise ValueError("fault injection unexpectedly committed")
        calls+=1;partial=rpc(request(proof,run_id,"fault_short","fault_readback"),cache,deadline)
        if not partial.get("second_file_absent") or not partial.get("commit_absent") or partial.get("same_target")!=record["same_target"]:
            raise ValueError("fault partial file/commit absence proof")
        record["partial_readback"]=partial
        for item in proof["boundaries"]:
            calls+=1;ack=rpc(request(proof,run_id,"fault_short","fallback_file",[item["name"]]),cache,deadline)
            receipt=dict(path=str(target_root(run_id,"fault_short",proof["fixture"])/item["name"]),bytes=item["bytes"],sha256=item["sha256"])
            if ack["publications"]!=[dict(path=receipt["path"],sha256=item["sha256"],receipt=receipt)]:
                raise ValueError("fallback original complete sameSHA production receipt")
        calls+=1;done=rpc(request(proof,run_id,"fault_short","finalize"),cache,deadline)
        calls+=1;checked=rpc(request(proof,run_id,"fault_short","readback"),cache,deadline)
        if done["commit"]!=checked["commit"] or done["publications"]!=checked["publications"] or not checked["all_file_tar_commit_readback"]:
            raise ValueError("same-target fallback complete files/tar/commit expectedSHA")
        record.update(complete=True,commit=checked["commit"],all_file_tar_commit_readback=True,publication_only_same_payload_same_target=True)
    except Exception as error:
        record.update(complete=False,error=type(error).__name__+": "+str(error),no_retry=True)
    record.update(actual_rpc_calls=calls,wall_seconds=time.monotonic()-start)
    local_artifact(run_id,"topology_fault_fallback.json",record)
    return record

def topology_run(run_id, fixture):
    # A single immutable attempt marker prevents another A/B run after any failure.
    marker = OWN / "TOPOLOGY_ATTEMPT.json"
    parent_fd = directory_fd(OWN)
    try:
        marker_fd = os.open(marker.name, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600, dir_fd=parent_fd)
        with os.fdopen(marker_fd, "wb") as out:
            out.write(encoded(dict(test_only=True,run_id=run_id,fixture=fixture,no_AB_retry=True)));out.flush();os.fsync(out.fileno())
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    started = time.monotonic()
    overall = started+1200
    # Worker04 needs no central mount: setup proof and original archive come through exactly one RPC.
    cache = local_cache(run_id, fixture)
    setup_start = time.monotonic()
    try:
        prepared = rpc(prepare_request(run_id,fixture),cache,min(overall,setup_start+600),prepare=True)
    except Exception as error:
        result=dict(test_only=True,run_id=run_id,fixture=fixture,complete=False,stage="setup",
                    actual_total_rpc_calls=RPC_PROCESSES_STARTED,error=type(error).__name__+": "+str(error),
                    no_AB_retry=True,model_calls=0,sourceclaims=0,actual_fullpipeline_speedup=False,
                    totalwall_seconds=time.monotonic()-started)
        local_artifact(run_id,"topology_result.json",result);print(json.dumps(result),flush=True);return result
    proof = prepared["proof"]
    result = dict(test_only=True,no_annotation=True,newsource=0,model_calls=0,sourceclaims=0,actual_fullpipeline_speedup=False,
                  run_id=run_id,fixture=fixture,topology="a100-extra-04 -> a100-extra-03 existing strict approved SSH",
                  setup_rpc_calls=1,setupwall_seconds=time.monotonic()-setup_start,arms=[],source_count=20,
                  original_commit_sha256=proof["original_commit_sha256"],original_archive_sha256=proof["original_commit"]["batch_sha256"],
                  archive_bytes=proof["boundaries"][-1]["bytes"],case_files=len(proof["boundaries"])-1)
    local_artifact(run_id,"topology_setup.json",result)
    for arm in ("baseline","batch"):
        start=time.monotonic(); deadline=min(overall,start+600); calls=0
        try:
            if arm=="baseline":
                for item in proof["boundaries"]:
                    calls+=1;ack=rpc(request(proof,run_id,arm,"file",[item["name"]]),cache,deadline)
                    expected=dict(path=str(target_root(run_id,arm,fixture)/item["name"]),bytes=item["bytes"],sha256=item["sha256"])
                    if ack["publications"] != [dict(path=expected["path"],sha256=item["sha256"],receipt=expected)]:
                        raise ValueError("baseline full exact receipt")
                calls+=1;ack=rpc(request(proof,run_id,arm,"finalize"),cache,deadline)
            else:
                calls+=1;ack=rpc(request(proof,run_id,arm,"batch"),cache,deadline)
            publishwall=time.monotonic()-start
            calls+=1;checked=rpc(request(proof,run_id,arm,"readback"),cache,deadline)
            if ack["commit"]!=checked["commit"] or ack["publications"]!=checked["publications"] or not checked["all_file_tar_commit_readback"]:
                raise ValueError("all file/tar/commit exact receiver FD readback")
            record=dict(arm=arm,complete=True,publishwall_seconds=publishwall,wall_seconds=time.monotonic()-start,
                        publication_rpc_calls=calls-1,readback_rpc_calls=1,actual_rpc_calls=calls,commit=checked["commit"],
                        all_file_tar_commit_readback=True,source_count=20,case_files=len(proof["boundaries"])-1)
            result["arms"].append(record);local_artifact(run_id,arm+"_result.json",record)
        except Exception as error:
            result["arms"].append(dict(arm=arm,complete=False,actual_rpc_calls=calls,error=type(error).__name__+": "+str(error),no_retry=True))
            result["totalwall_seconds"]=time.monotonic()-started
            result["actual_total_rpc_calls"]=RPC_PROCESSES_STARTED
            local_artifact(run_id,"topology_result.json",result)
            print(json.dumps(result),flush=True);return result
    result["fault_fallback"]=topology_fault_fallback(proof,cache,run_id,min(overall,time.monotonic()+600))
    result["actual_total_rpc_calls"]=RPC_PROCESSES_STARTED
    result["totalwall_seconds"]=time.monotonic()-started
    result["publication_only_observed_wall_ratio"]=result["arms"][0]["publishwall_seconds"]/result["arms"][1]["publishwall_seconds"]
    local_artifact(run_id,"topology_result.json",result)
    print(json.dumps(result),flush=True)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--cpu-contracts",action="store_true")
    modes.add_argument("--test-only-topology",action="store_true")
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--fixture",choices=tuple(FIXTURES),default="mixed")
    args=parser.parse_args()
    run_root(args.run_id)
    if args.cpu_contracts:
        from contract_tests import run_tests
        print(json.dumps(run_tests(args.run_id)),flush=True)
    else:
        if not (OWN/"READY.json").is_file():
            parser.error("fixed frozen READY required before topology")
        ready=strict(bounded_read(OWN/"READY.json",HEADER_CAP))
        for name, expected in ready["source_sha256"].items():
            bounded_read(OWN/name,HEADER_CAP,expected)
        topology_run(args.run_id,args.fixture)

if __name__=="__main__":
    main()
