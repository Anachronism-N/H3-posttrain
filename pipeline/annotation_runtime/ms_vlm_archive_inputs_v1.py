"""Bounded CPU-only raw archive metadata bridge; never rebind original JSON."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
ROOT = RT.parent
sys.path.insert(0, str(RT))
import ms_fullframe_vlm_postprocess_v3 as v

MAX_ARCHIVE_BYTES = 128 * 2**20
MAX_JSON_BYTES = 2**20
OUTPUT = ROOT / "ms_fullframe_vlm_archive_inputs_20261027_v1"
SUCCESS = "extracted_rule_draft_unapproved"
_VIDEO_PARENT_CACHE = {}


def need(ok, text):
    v.need(ok, text)


def encoded(value):
    return v.h.bench.encoded(value)


def raw_ref(path, raw):
    return dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(raw).hexdigest())


def write_raw_once(path, raw):
    path = Path(path)
    need(path.is_absolute() and str(path.resolve()) == str(path), "Stable absolute raw output path required")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            need(stat.S_ISREG(os.fstat(stream.fileno()).st_mode) and stream.read(len(raw) + 1) == raw,
                 "Existing materialization differs; no overwrite")
        return raw_ref(path, raw)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    need(path.read_bytes() == raw, "Durable raw readback differs")
    return raw_ref(path, raw)


def json_once(path, value):
    return write_raw_once(path, encoded(value))


def signature(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def safe_member(member):
    name = member.name
    path = PurePosixPath(name)
    need(name and not path.is_absolute() and str(path) == name and
         all(p not in ("", ".", "..") for p in name.split("/")) and "\\" not in name,
         "Unsafe archive traversal/member name")
    need(member.isfile() and not member.issparse() and member.sparse is None and
         not any(k.startswith("GNU.sparse") or k.startswith("SCHILY.realsize") for k in member.pax_headers),
         "Only regular nonsparse archive members allowed")
    need(type(member.size) is int and 0 <= member.size <= MAX_ARCHIVE_BYTES, "Archive member exceeds bound")


class RegularTarInfo(tarfile.TarInfo):
    def _proc_sparse(self, *args):
        raise ValueError("GNU sparse archive forbidden before payload parsing")

    def _proc_gnusparse_00(self, *args):
        raise ValueError("PAX sparse archive forbidden before payload parsing")

    def _proc_gnusparse_01(self, *args):
        raise ValueError("PAX sparse archive forbidden before payload parsing")

    def _proc_gnusparse_10(self, *args):
        raise ValueError("PAX sparse archive forbidden before payload parsing")


def read_archive(commit_ref, commit, wanted, output, cache):
    path = Path(commit["archive_path"])
    need(path.is_absolute() and str(path.resolve(strict=True)) == str(path), "Stable original archive path required")
    need(re.fullmatch(r"[a-f0-9]{64}", commit.get("archive_sha256", "")) and
         type(commit.get("archive_bytes")) is int and 0 < commit["archive_bytes"] <= MAX_ARCHIVE_BYTES,
         "Finite original archive SHA/bytes required")
    key = str(path)
    if key in cache:
        record = cache[key]
        need(record["commit_ref"] == commit_ref and set(wanted) <= record["wanted"] and
             signature(path.stat()) == record["signature"], "Cached archive binding/stat differs")
        return record
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_size == commit["archive_bytes"], "Original archive bytes/type differ")
        raw = stream.read(MAX_ARCHIVE_BYTES + 1)
        after = os.fstat(stream.fileno())
    need(signature(before) == signature(after) == signature(path.stat()) and len(raw) == before.st_size and
         hashlib.sha256(raw).hexdigest() == commit["archive_sha256"], "Whole archive SHA or stable stat differs")
    if "batch_sha256" in commit:
        need(commit["batch_sha256"] == commit["archive_sha256"], "Original batch SHA differs")
    extracted, names = {}, set()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:", tarinfo=RegularTarInfo) as archive:
        for member in archive:
            safe_member(member)
            need(member.name not in names, "Duplicate archive member forbidden")
            names.add(member.name)
            if member.name in wanted:
                need(member.size <= MAX_JSON_BYTES, "Raw JSON member exceeds 1 MiB bound")
                need(member.name.endswith(("/draft.json", "/fullframe_manifest.json")), "Only draft/manifest extraction allowed")
                stream = archive.extractfile(member)
                value = stream.read(MAX_JSON_BYTES + 1)
                need(len(value) == member.size, "Original archive member bytes differ")
                v.h.bench.strict(value)
                extracted[member.name] = value
    need(set(extracted) == set(wanted), "Selected original JSON members missing")
    verification = dict(schema="ms_archive_vlm_batch_verified_v1", commit_ref=commit_ref,
        archive_path=str(path), archive_sha256=commit["archive_sha256"], archive_bytes=commit["archive_bytes"],
        sha_verified_once=True, whole_archive_hash_reads=1, archive_stat_stable=True,
        archive_stat=list(signature(before)), archive_member_count=len(names),
        hash_includes_npz_packaged_bytes=True, no_npz_decoded=True, no_media_input_read=True, no_gpu_calls=True)
    vr = json_once(Path(output) / "batch_proofs" / (hashlib.sha256(str(path).encode()).hexdigest() + ".json"), verification)
    result = dict(commit_ref=commit_ref, wanted=set(wanted), signature=signature(before),
                  extracted=extracted, names=names, verification=verification, verification_ref=vr)
    cache[key] = result
    return result


def eligible(source, receipt, excluded, seen):
    need(isinstance(source.get("id"), str) and re.fullmatch(r"[A-Za-z0-9_-]+", source["id"]), "Unsafe exact source ID")
    need(source.get("split") == "train", "Only original train may enter this train plan")
    need(source["id"].casefold() not in excluded, "Historical old500 source excluded")
    need(source["id"].casefold() not in seen, "Casefold duplicate source rejected")
    need(receipt.get("source_row") == source and receipt.get("status") == SUCCESS and
         all(receipt.get(k) is value for k, value in list(v.FLAGS.items())[:4]),
         "Failed/oldclaim/unapproved receipt is not a successful train source")
    video = Path(source["video_path"])
    parent = video.parent
    if parent not in _VIDEO_PARENT_CACHE:
        need(parent.is_absolute() and parent.resolve() == parent, "Stable original video parent required")
        _VIDEO_PARENT_CACHE[parent] = signature(parent.stat())[:2]
    need(signature(parent.stat())[:2] == _VIDEO_PARENT_CACHE[parent] and video.is_absolute() and
         str(video) == source["video_path"] and video.stem == source["id"] and not video.is_symlink(),
         "Original video identity/path differs")


def materialize(commit_ref, commit, ids, output, cache, excluded=None, seen=None):
    need(commit.get("schema") == "ms_unified_novlm_batch_commit_v3", "Exact original archive commit schema required")
    excluded = excluded if excluded is not None else set()
    seen = seen if seen is not None else set()
    wanted = set()
    for id_ in ids:
        item = commit["archive_index"][id_]
        receipts = [r for r in commit["receipts"] if r.get("source_row", {}).get("id") == id_]
        need(len(receipts) == 1, "Exact original receipt required")
        eligible(item["source_row"], receipts[0], excluded, seen)
        wanted.update(item["members"][n] for n in ("draft.json", "fullframe_manifest.json"))
    batch = read_archive(commit_ref, commit, wanted, output, cache)
    rows = []
    for id_ in ids:
        item = commit["archive_index"][id_]
        source, members = item["source_row"], item["members"]
        need(set(members.values()) <= batch["names"], "Original committed members missing")
        receipt = next(r for r in commit["receipts"] if r["source_row"]["id"] == id_)
        pair = {name: batch["extracted"][members[name]] for name in ("draft.json", "fullframe_manifest.json")}
        directory = Path(output) / "sources" / source["split"] / id_
        refs = {name: write_raw_once(directory / name, raw) for name, raw in pair.items()}
        draft, manifest = (v.h.bench.strict(pair[n]) for n in ("draft.json", "fullframe_manifest.json"))
        origin = str(Path(commit_ref["path"]).parent / members["fullframe_manifest.json"])
        proof = dict(schema="ms_archive_vlm_input_origin_v1", commit_ref=commit_ref, source_row=source,
            members=members, source_case_prefix=item["source_case_prefix"], receipt=receipt,
            archive_path=commit["archive_path"], archive_sha256=commit["archive_sha256"], archive_bytes=commit["archive_bytes"],
            sha_verified_once=True, archive_verification_ref=batch["verification_ref"],
            member_metadata={name: dict(member=members[name], sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)) for name, raw in pair.items()},
            physical_refs=refs, logical_fullframe_origin=origin,
            original_binding=draft.get("source", {}).get("fullframe_binding"), original_npz_path=manifest.get("npz_path"),
            no_npz_decoded=True, no_media_input_read=True, no_gpu_calls=True,
            original_raw_bytes_preserved=True, hash_includes_npz_packaged_bytes=True)
        proof_ref = json_once(directory / "origin_proof.json", proof)
        row = dict(id=id_, split=source["split"], video_path=source["video_path"],
                   rule_draft_ref=refs["draft.json"], fullframe_manifest_ref=refs["fullframe_manifest.json"],
                   fullframe_origin_path=origin, origin_proof_ref=proof_ref)
        v.input_pair(row)
        need(id_.casefold() not in seen, "Duplicate batch source rejected")
        seen.add(id_.casefold())
        rows.append(row)
    return rows


def historical_exclusion(root):
    registry_path = root / "runtime/ms_vlm_postprocess_jobs_registry_v1.json"
    user_path = root / "ms_fullframe_vlm_postprocess_v2_user200_20261027_allocation.json"
    registry, allocation = v.load(registry_path), v.load(user_path)
    refs = [v.ref(registry_path), v.ref(user_path)]
    old_rows = []
    oldjobs = [j for j in registry["jobs"] if "_user200_" not in j["phase"]]
    need(len(oldjobs) == 8 and sum(j["sources"] for j in oldjobs) == 300, "All actual eight oldjobs/300 required")
    for job in oldjobs:
        cfg = v.load(job["config_path"])
        refs.extend([v.ref(job["config_path"]), cfg["plan"]])
        need(v.base.sha(cfg["plan"]["path"]) == cfg["plan"]["sha256"], "Historical frozen plan changed")
        rows = [v.h.bench.strict(line) for line in Path(cfg["plan"]["path"]).read_bytes().splitlines() if line.strip()]
        need(len(rows) == job["sources"] and [r["id"] for r in rows] == cfg["selected_source_ids"], "Historical selected IDs differ")
        old_rows.extend((r["split"], r["id"]) for r in rows)
    user_rows = []
    need(allocation["total_sources"] == 200 and len(allocation["allocation"]) == 3, "All frozen user200 plans required")
    for lane in allocation["allocation"]:
        cfg = v.checked(lane["config_ref"])
        need(cfg["plan"] == lane["plan_ref"], "User200 frozen config/plan differs")
        refs.extend([lane["config_ref"], lane["plan_ref"]])
        need(v.base.sha(lane["plan_ref"]["path"]) == lane["plan_ref"]["sha256"], "User200 frozen plan differs")
        rows = [v.h.bench.strict(line) for line in Path(lane["plan_ref"]["path"]).read_bytes().splitlines() if line.strip()]
        need([r["id"] for r in rows] == cfg["selected_source_ids"] and len(rows) == cfg["targetcount"], "User200 exact rows differ")
        user_rows.extend((r["split"], r["id"]) for r in rows)
    need(len(user_rows) == 200 and sorted(map(list, user_rows)) == allocation["source_keys"], "Frozen full200 source_keys differ")
    keys = old_rows + user_rows
    need(len(keys) == len(set(keys)) == len({id_.casefold() for _, id_ in keys}) == 500, "All old500 exact/casefold uniqueness required")
    return {id_.casefold() for _, id_ in keys}, keys, refs


def select_jobs(root, excluded, count):
    allocation_path = root / "ms_unified_continuous_allocation_20261027_v1/allocation.json"
    allocation = v.load(allocation_path)
    need(len(allocation["jobs"]) == 8, "Exact frozen eight-stream allocation required")
    refs, selected, source_counts, seen = [v.ref(allocation_path)], [], {}, set()
    for job in allocation["jobs"][:4]:
        if len(seen) == count:
            break
        phase = job["phase"]
        cfg_path, summary_path = root / "runtime" / (phase + "_config.json"), root / phase / "summary.json"
        cfg, summary = v.load(cfg_path), v.load(summary_path)
        ticket = v.checked(cfg["ticket"])
        need(cfg["phase"] == summary["edition"] == ticket["edition"] == phase and summary.get("status") == "completed" and
             cfg["plan"] == dict(path=job["plan"], sha256=job["plan_sha256"]) and
             cfg["targetcount"] == cfg["input_count"] == job["source_count"] == 1000 and
             summary["processed"] == 1000 and summary["selected_source_ids"] == cfg["selected_source_ids"],
             "Exact completed source job/config/ticket/summary binding required")
        need(v.base.sha(job["plan"]) == job["plan_sha256"], "Original source job plan SHA differs")
        plan = [v.h.bench.strict(line) for line in Path(job["plan"]).read_bytes().splitlines() if line.strip()]
        need(plan == ticket["sources"] and [s["id"] for s in plan] == cfg["selected_source_ids"], "Original ticket/plan sources differ")
        planmap = {s["id"]: s for s in plan}
        outcomes = {(s["batch_index"], s["id"]): s for s in summary["all_outcomes"]}
        need(len(outcomes) == len(summary["all_outcomes"]) == 1000, "Unique exact completed source outcomes required")
        refs.extend([v.ref(cfg_path), v.ref(summary_path), cfg["ticket"], cfg["plan"]])
        source_counts[phase] = 0
        for batch in summary["completed_batches"]:
            commit_ref = dict(path=batch["commit_path"], sha256=batch["commit_sha256"])
            commit = v.frozen_commit(commit_ref)
            need(commit["schema"] == "ms_unified_novlm_batch_commit_v3" and commit["edition"] == phase and
                 commit["parent_ticket_id"] == ticket["ticket_id"] and commit["batch_index"] == batch["batch_index"] and
                 commit["ticket_id"] == ticket["ticket_id"] + "-b%06d" % batch["batch_index"], "Original successful commit/ticket binding differs")
            ids = []
            for receipt in commit["receipts"]:
                source = receipt["source_row"]
                id_ = source["id"]
                need(source == planmap.get(id_) and commit["archive_index"][id_]["source_row"] == source,
                     "Original committed source row differs from ticket/plan")
                outcome = outcomes.get((batch["batch_index"], id_))
                need(outcome and outcome["split"] == source["split"] and outcome["status"] == receipt["status"], "Summary/commit success receipt differs")
                if receipt["status"] != SUCCESS or source["split"] != "train" or id_.casefold() in excluded:
                    continue
                eligible(source, receipt, excluded, seen)
                ids.append(id_)
                seen.add(id_.casefold())
                source_counts[phase] += 1
                if len(seen) == count:
                    break
            if ids:
                refs.append(commit_ref)
                selected.append((commit_ref, commit, ids))
            if len(seen) == count:
                break
    need(len(seen) == count, "At most four explicit completed summaries lack requested unique train sources")
    return selected, source_counts, refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=3000)
    args = ap.parse_args()
    need(args.count == 3000, "Authorized exact 3000train finite materialization only")
    excluded, oldkeys, oldrefs = historical_exclusion(ROOT)
    selected, source_counts, source_refs = select_jobs(ROOT, excluded, args.count)
    rows, cache, seen = [], {}, set()
    first_ref, first_commit, first_ids = selected[0]
    wanted = {first_commit["archive_index"][id_]["members"][name]
              for id_ in first_ids for name in ("draft.json", "fullframe_manifest.json")}
    read_archive(first_ref, first_commit, wanted, OUTPUT, cache)
    pilot_ids = first_ids[:3]
    need(len(pilot_ids) == 3, "Exactly three real bridge pilot sources required")
    rows.extend(materialize(first_ref, first_commit, pilot_ids, OUTPUT, cache, excluded, seen))
    print(json.dumps(dict(real_source_cpu_input_pair_pass=3, ids=pilot_ids, no_gpu_calls=True)), flush=True)
    for commit_ref, commit, ids in selected:
        remaining = [id_ for id_ in ids if id_.casefold() not in seen]
        if remaining:
            rows.extend(materialize(commit_ref, commit, remaining, OUTPUT, cache, excluded, seen))
    need(len(rows) == len(seen) == 3000 and not seen & excluded and all(row["split"] == "train" for row in rows),
         "Exact finite unique3000train/old500 disjointness required")
    manifest_ref = write_raw_once(OUTPUT / "manifest.jsonl", b"".join(encoded(row) for row in rows))
    keys = [[row["split"], row["id"]] for row in rows]
    proof = dict(schema="ms_archive_vlm_inputs_allocation_v1", status="frozen_cpu_materialization_only", total_sources=len(rows),
        manifest_ref=manifest_ref, source_keys=keys, source_keys_sha256=hashlib.sha256(encoded(keys)).hexdigest(),
        unique_exact_ids=len({row["id"] for row in rows}), unique_casefold_ids=len(seen), splits=dict(train=len(rows)),
        historical_exclusion_count=len(oldkeys), historical_source_keys_sha256=hashlib.sha256(encoded(sorted(map(list, oldkeys)))).hexdigest(),
        historical_overlap_exact=0, historical_overlap_casefold=0, original_job_source_counts=source_counts,
        original_commit_count=len(selected), whole_archive_hash_reads=len(cache),
        whole_archive_hash_bytes=sum(batch["verification"]["archive_bytes"] for batch in cache.values()),
        original_refs=source_refs, historical_refs=oldrefs,
        archive_verification_refs=[batch["verification_ref"] for batch in cache.values()],
        all_json_refs=[row[key] for row in rows for key in ("rule_draft_ref", "fullframe_manifest_ref", "origin_proof_ref")],
        code_refs=[v.ref(__file__), v.ref(v.SELF)], real_cpu_pilot_sources=pilot_ids, cpu_input_pair_pass_count=len(rows),
        no_gpu_calls=True, no_model_load=True, no_npz_decoded=True, no_media_input_read=True,
        hash_includes_npz_packaged_bytes=True, original_raw_bytes_preserved=True, original_binding_digests_unchanged=True,
        no_running_configs_or_queues_created=True, **v.FLAGS)
    allocation_ref = json_once(OUTPUT / "allocation.json", proof)
    print(json.dumps(dict(allocation_ref=allocation_ref, manifest_ref=manifest_ref, count=len(rows),
        original_job_source_counts=source_counts, archives_hashed_once=len(cache), old500_overlap=0, no_gpu_calls=True), sort_keys=True))


if __name__ == "__main__":
    main()
