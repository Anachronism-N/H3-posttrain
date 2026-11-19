"""Frozen-selection, missing-only CPU metadata recovery; no production stages."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
sys.path.insert(0, str(RT))
import ms_vlm_archive_inputs_v1 as a
v = a.v
ROOT = RT.parent
POOL = ROOT / "ms_fullframe_vlm_archive_inputs_20261027_v1"
PREVIOUS = ROOT / "ms_fullframe_vlm_archive_inputs_recovery_20261027_v1"
OUTPUT = ROOT / "ms_fullframe_vlm_archive_inputs_recovery_20261028_v2"
SELECTION_SHA = "d5edc46b0782a3c44837e72efcc555c3c46b05f882ecb05ba9011d904dca3d56"
RECOVERY_SHA = "b8de72cd8b7d7cb1ffbf26901a8439987ce458bb408fff9dcbf061888ff51649"
MEMBERS = {"draft.json", "fullframe_manifest.json", "pose.npz", "receipt.json", "beats.json"}


def once(path, raw):
    path = Path(path)
    v.need(path.is_absolute() and path.resolve() == path, "Stable absolute evidence path required")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    v.need(path.read_bytes() == raw, "Write-once evidence readback differs")
    return a.raw_ref(path, raw)


def proof_path(output, archive):
    return output / "batch_proofs" / (hashlib.sha256(str(archive).encode()).hexdigest() + ".json")


def verified_batch(commit_ref, commit, path):
    reference = v.ref(path)
    value = v.checked(reference)
    v.need(value.get("schema") == "ms_archive_vlm_batch_verified_v1" and
           value.get("commit_ref") == commit_ref and
           all(value.get(k) == commit[k] for k in ("archive_path", "archive_sha256", "archive_bytes")) and
           value.get("whole_archive_hash_reads") == 1 and value.get("sha_verified_once") is True and
           value.get("archive_stat_stable") is True and
           all(value.get(k) is True for k in ("no_gpu_calls", "no_npz_decoded", "no_media_input_read")),
           "Existing once-only archive proof binding differs")
    v.need(list(a.signature(Path(commit["archive_path"]).stat())) == value.get("archive_stat"),
           "Previously verified archive stat changed; never silently rehash")
    return reference, value


def read_verified_members(commit_ref, commit, wanted, path):
    reference, verification = verified_batch(commit_ref, commit, path)
    archivepath = Path(commit["archive_path"])
    fd = os.open(archivepath, os.O_RDONLY | os.O_NOFOLLOW)
    names, extracted = set(), {}
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        v.need(stat.S_ISREG(before.st_mode) and list(a.signature(before)) == verification["archive_stat"],
               "Verified archive descriptor stat differs")
        with tarfile.open(fileobj=stream, mode="r:", tarinfo=a.RegularTarInfo) as archive:
            for member in archive:
                a.safe_member(member)
                v.need(member.name not in names, "Duplicate original member")
                names.add(member.name)
                if member.name in wanted:
                    v.need(member.size <= a.MAX_JSON_BYTES and member.name.endswith(("/draft.json", "/fullframe_manifest.json")),
                           "Only bounded raw JSON may be recovered")
                    raw = archive.extractfile(member).read(a.MAX_JSON_BYTES + 1)
                    v.need(len(raw) == member.size, "Original raw member length differs")
                    v.h.bench.strict(raw)
                    extracted[member.name] = raw
        after = os.fstat(stream.fileno())
    v.need(a.signature(before) == a.signature(after) == a.signature(archivepath.stat()) and
           set(extracted) == set(wanted) and len(names) == verification["archive_member_count"],
           "Cached stat/member verification differs")
    return dict(commit_ref=commit_ref, verification_ref=reference, verification=verification,
                extracted=extracted, names=names, signature=a.signature(before), wanted=set(wanted))


def strict_archive(row, draft, manifest, proof, commit):
    item = commit["archive_index"][row["id"]]
    source, members = item["source_row"], item["members"]
    prefix = "cases/" + row["split"] + "/" + row["id"]
    v.need(set(members) == MEMBERS and all(members[n] == prefix + "/" + n for n in MEMBERS),
           "All five exact archive members required")
    locator = dict(archive_path=commit["archive_path"], source_case_prefix=prefix,
                   fullframe_manifest_member=members["fullframe_manifest.json"], npz_member=members["pose.npz"])
    receipt = proof["receipt"]
    for value in (draft["source"], manifest, receipt):
        v.need(value.get("locator") == locator and value.get("artifact_storage") == "archive_member",
               "Three mandatory locators and storage bindings required")
    v.need(draft["source"].get("artifact_members") == receipt.get("artifact_members") == members and
           item.get("receipt_member") == members["receipt.json"] and proof.get("members") == members,
           "Original member bindings differ")
    virtual_npz = str(Path(proof["commit_ref"]["path"]).parent / members["pose.npz"])
    v.need(draft["source"].get("npz_path") == manifest.get("npz_path") == proof.get("original_npz_path") == virtual_npz,
           "Original virtual NPZ path differs")
    v.need(source["id"] == row["id"] and source["split"] == row["split"] == "train" and
           source["video_path"] == row["video_path"], "Original train source binding differs")


def check_row(row, commit_ref, commit):
    draft, manifest = v.input_pair(row)
    proof = v.checked(row["origin_proof_ref"])
    v.need(proof["commit_ref"] == commit_ref, "Row belongs to another frozen commit")
    strict_archive(row, draft, manifest, proof, commit)
    return row


def original_row(id_, commit_ref, commit, directory):
    path = directory / "origin_proof.json"
    proof = v.load(path)
    refs = proof["physical_refs"]
    source = commit["archive_index"][id_]["source_row"]
    row = dict(id=id_, split=source["split"], video_path=source["video_path"],
               rule_draft_ref=refs["draft.json"], fullframe_manifest_ref=refs["fullframe_manifest.json"],
               origin_proof_ref=v.ref(path), fullframe_origin_path=proof["logical_fullframe_origin"])
    return check_row(row, commit_ref, commit)


def recover_row(id_, commit_ref, commit, batch, output, original):
    item = commit["archive_index"][id_]
    source, members = item["source_row"], item["members"]
    receipt = next(r for r in commit["receipts"] if r["source_row"]["id"] == id_)
    a.eligible(source, receipt, set(), set())
    v.need(set(members.values()) <= batch["names"], "Original committed members missing")
    rawpair = {n: batch["extracted"][members[n]] for n in ("draft.json", "fullframe_manifest.json")}
    refs, preserved_zero = {}, []
    for name, raw in rawpair.items():
        old = original / name
        if old.exists() and old.stat().st_size:
            v.need(not old.is_symlink() and old.read_bytes() == raw, "Existing raw original differs; never overwrite")
            refs[name] = a.raw_ref(old, raw)
        else:
            if old.exists(): preserved_zero.append(dict(path=str(old), bytes=0))
            refs[name] = once(output / "sources" / source["split"] / id_ / name, raw)
    oldproof = original / "origin_proof.json"
    if oldproof.exists():
        v.need(not oldproof.is_symlink() and oldproof.stat().st_size == 0, "Only missing or zero-byte proof may be recovered")
        preserved_zero.append(dict(path=str(oldproof), bytes=0))
    draft, manifest = (v.h.bench.strict(rawpair[n]) for n in ("draft.json", "fullframe_manifest.json"))
    origin = str(Path(commit_ref["path"]).parent / members["fullframe_manifest.json"])
    proof = dict(schema="ms_archive_vlm_input_origin_v1", commit_ref=commit_ref, source_row=source,
        members=members, source_case_prefix=item["source_case_prefix"], receipt=receipt,
        archive_path=commit["archive_path"], archive_sha256=commit["archive_sha256"], archive_bytes=commit["archive_bytes"],
        sha_verified_once=True, archive_verification_ref=batch["verification_ref"],
        member_metadata={n: dict(member=members[n], sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)) for n, raw in rawpair.items()},
        physical_refs=refs, logical_fullframe_origin=origin,
        original_binding=draft["source"]["fullframe_binding"], original_npz_path=manifest["npz_path"],
        no_npz_decoded=True, no_media_input_read=True, no_gpu_calls=True,
        original_raw_bytes_preserved=True, hash_includes_npz_packaged_bytes=True)
    row = dict(id=id_, split=source["split"], video_path=source["video_path"], rule_draft_ref=refs["draft.json"],
               fullframe_manifest_ref=refs["fullframe_manifest.json"], fullframe_origin_path=origin,
               origin_proof_ref=once(output / "sources" / source["split"] / id_ / "origin_proof.json", a.encoded(proof)))
    return check_row(row, commit_ref, commit), preserved_zero


def allocations():
    path = ROOT / "runtime/ms_vlm_postprocess_jobs_registry_v1.json"
    raw = path.read_bytes(); registry = v.h.bench.strict(raw)
    keys, refs, configs, lanes = set(), [], [], []
    for job in registry["jobs"]:
        cfg_ref = v.ref(job["config_path"]); cfg = v.checked(cfg_ref); configs.append(cfg_ref)
        plan_ref = cfg["plan"]; plan_raw = Path(plan_ref["path"]).read_bytes()
        v.need(hashlib.sha256(plan_raw).hexdigest() == plan_ref["sha256"], "Allocated frozen plan changed")
        rows = [v.h.bench.strict(line) for line in plan_raw.splitlines() if line.strip()]
        v.need(len(rows) == job["sources"] and [r["id"] for r in rows] == cfg["selected_source_ids"], "Allocated source IDs differ")
        keys.update(r["id"].casefold() for r in rows); refs.append(plan_ref)
        if "_archive250_" in job["phase"]:
            lanes.append(dict(phase=job["phase"], plan_ref=plan_ref, source_keys=[[r["split"], r["id"]] for r in rows]))
    v.need(len(lanes) == 4 and all(len(x["source_keys"]) == 250 for x in lanes) and
           len({i.casefold() for lane in lanes for _, i in lane["source_keys"]}) == 1000,
           "Four disjoint preallocated250 plans required")
    return dict(registry_ref=a.raw_ref(path, raw), config_refs=configs, plan_refs=refs, allocated_casefold_ids=keys,
                archive1000_plans=lanes, jobs_count=len(registry["jobs"]))


def main():
    selection_ref = dict(path=str(POOL / "source_selection_proof.json"), sha256=SELECTION_SHA)
    selection = v.checked(selection_ref)
    cpu = v.checked(selection["frozen_cpu_proof_ref"])
    for path, digest in cpu["source_sha256"].items():
        v.need(v.base.sha(path) == digest, "Frozen original code changed: " + path)
    keys = selection["source_keys"]
    v.need(len(keys) == 3000 and len({i.casefold() for _, i in keys}) == 3000 and
           hashlib.sha256(a.encoded(keys)).hexdigest() == selection["source_keys_sha256"] and
           all(s == "train" for s, _ in keys) and len(selection["selected_batches"]) == 150,
           "Exact frozen3000 selection required")
    allocated = allocations()
    receipt_ref = dict(path=str(PREVIOUS / "recovery_receipt.json"), sha256=RECOVERY_SHA)
    prior = v.checked(receipt_ref)
    previous_row = prior["restored_row"]
    v.need(Path(prior["preserved_incomplete_file"]).stat().st_size == 0, "Historical zero-byte original changed")
    existing_names = {p.name for p in (POOL / "sources/train").iterdir()}
    old_proofs = {p.name for p in (POOL / "batch_proofs").iterdir()}
    v.need(not OUTPUT.exists(), "Independent output already exists; inspect instead of duplicate restart")
    OUTPUT.mkdir()
    code_ref = once(OUTPUT / "frozen_recovery_code.py", Path(__file__).read_bytes())
    accepted_cpu = v.load(RT / "ms_vlm_archive_inputs_recovery_v2_cpu_proof.json")
    v.need(accepted_cpu["code_sha256"] == code_ref["sha256"] and accepted_cpu["exitcode"] == 0,
           "Recovery code must pass CPU tests before materialization")
    once(OUTPUT / "cpu_proof.json", a.encoded(accepted_cpu))

    def process(entry):
        commit_ref, ids = entry["commit_ref"], entry["source_ids"]
        commit = v.frozen_commit(commit_ref)
        v.need(commit["schema"] == "ms_unified_novlm_batch_commit_v3", "Frozen archive commit schema differs")
        rows, missing, zeros = {}, [], []
        for id_ in ids:
            directory = POOL / "sources/train" / id_
            if id_ == previous_row["id"]:
                rows[id_] = check_row(previous_row, commit_ref, commit)
            elif id_ in existing_names and (directory / "origin_proof.json").exists() and (directory / "origin_proof.json").stat().st_size:
                rows[id_] = original_row(id_, commit_ref, commit, directory)
            else: missing.append(id_)
        old_path = proof_path(POOL, commit["archive_path"])
        reused = old_path.name in old_proofs
        new_hashes = 0
        if missing:
            wanted = {commit["archive_index"][id_]["members"][n] for id_ in missing for n in ("draft.json", "fullframe_manifest.json")}
            if reused: batch = read_verified_members(commit_ref, commit, wanted, old_path)
            else:
                batch = a.read_archive(commit_ref, commit, wanted, OUTPUT, {})
                new_hashes = 1
            for id_ in missing:
                row, preserved = recover_row(id_, commit_ref, commit, batch, OUTPUT, POOL / "sources/train" / id_)
                rows[id_] = row; zeros.extend(preserved)
            verification_ref = batch["verification_ref"]
        else:
            v.need(reused, "Complete originals require existing archive verification proof")
            verification_ref, _ = verified_batch(commit_ref, commit, old_path)
        evidence = dict(commit_ref=commit_ref, selected_source_ids=ids, rows=[rows[i] for i in ids],
                        original_rows_reused=len(ids)-len(missing), missing_rows_materialized=len(missing),
                        existing_archive_proof_reused=reused, new_whole_archive_hash_reads=new_hashes,
                        archive_verification_ref=verification_ref, preserved_zero_files=zeros)
        once(OUTPUT / "batch_receipts" / (hashlib.sha256(commit_ref["path"].encode()).hexdigest()+".json"), a.encoded(evidence))
        print(json.dumps(dict(batch=commit["ticket_id"], reused=len(ids)-len(missing), recovered=len(missing), first_hashes=new_hashes)), flush=True)
        return evidence

    with ThreadPoolExecutor(max_workers=4) as executor:
        batches = list(executor.map(process, selection["selected_batches"]))
    rows = [row for b in batches for row in b["rows"]]
    v.need([[r["split"], r["id"]] for r in rows] == keys, "Recovered manifest order differs from frozen selection")
    after = allocations()
    allocated_ids = allocated["allocated_casefold_ids"] | after["allocated_casefold_ids"]
    available = [r for r in rows if r["id"].casefold() not in allocated_ids]
    manifest_ref = once(OUTPUT / "manifest.jsonl", b"".join(a.encoded(r) for r in rows))
    available_ref = once(OUTPUT / "unallocated_manifest.jsonl", b"".join(a.encoded(r) for r in available))
    all_refs = [r[k] for r in rows for k in ("rule_draft_ref", "fullframe_manifest_ref", "origin_proof_ref")]
    proof = dict(schema="ms_archive_vlm_metadata_recovery_allocation_v2", status="frozen_cpu_materialization_only_not_deployed",
        source_count=3000, manifest_ref=manifest_ref, source_keys=keys, source_keys_sha256=selection["source_keys_sha256"],
        source_selection_ref=selection_ref, previous_recovery_receipt_ref=receipt_ref, code_ref=code_ref,
        cpu_input_pair_pass_count=3000, strict_locator_storage_member_preflight_pass_count=3000,
        reused_original_rows=sum(b["original_rows_reused"] for b in batches),
        missing_rows_materialized=sum(b["missing_rows_materialized"] for b in batches),
        reused_archive_proofs=sum(b["existing_archive_proof_reused"] for b in batches),
        new_once_only_archive_hash_reads=sum(b["new_whole_archive_hash_reads"] for b in batches),
        existing_archive_rehashes=0, archive_verification_refs=[b["archive_verification_ref"] for b in batches],
        preserved_zero_files=[z for b in batches for z in b["preserved_zero_files"]]+[dict(path=prior["preserved_incomplete_file"],bytes=0)],
        unallocated_manifest_ref=available_ref, unallocated_source_count=len(available),
        unallocated_plan_intersection=0, unallocated_observation_not_exclusive_reservation=True,
        registry_before_ref=allocated["registry_ref"], registry_after_ref=after["registry_ref"],
        allocated_plan_refs=after["plan_refs"], archive1000_plans=after["archive1000_plans"],
        all_json_refs=all_refs, source_claims_created=0, stage_budget_claims_created=0,
        run_configs_created=0, run_queues_created=0, models_called=0, no_gpu_calls=True, no_model_load=True,
        no_media_input_read=True, no_npz_decoded=True, original_raw_bytes_preserved=True,
        original_binding_digests_unchanged=True, original_files_overwritten=0, **v.FLAGS)
    allocation_ref = once(OUTPUT / "allocation.json", a.encoded(proof))
    print(json.dumps(dict(allocation_ref=allocation_ref, manifest_ref=manifest_ref,
                         unallocated_manifest_ref=available_ref, unallocated_source_count=len(available),
                         new_once_only_archive_hash_reads=proof["new_once_only_archive_hash_reads"],
                         existing_archive_rehashes=0, models_called=0),sort_keys=True),flush=True)


def self_test():
    tests = v.module("recovery_original_tests", RT / "test_ms_vlm_archive_inputs_v1.py")
    class RecoveryTests(unittest.TestCase):
        def setUp(self):
            self.fixture = tests.ArchiveInputTests("test_rawbytes_logical_origin_binding_and_once_only_hash")
            self.fixture.setUp(); f=self.fixture
            f.members["beats.json"] = f.prefix + "/beats.json"
            f.item["receipt_member"] = f.members["receipt.json"]
            for value in (f.draft["source"], f.manifest, f.receipt): value["artifact_storage"] = "archive_member"
            f.draft["source"]["artifact_members"] = f.members
            f.draft["source"]["npz_path"] = f.manifest["npz_path"]
            f.rebind()
            extra=tarfile.TarInfo(f.members["beats.json"]); extra.size=2
            f.create(extras=[(extra,b"{}")])
            self.row=f.bridge(); self.proof=v.checked(self.row["origin_proof_ref"])
        def tearDown(self): self.fixture.tearDown()
        def test_strict_positive(self): check_row(self.row,self.fixture.commit_ref,self.fixture.commit)
        def test_mandatory_locators(self):
            for target in ("draft", "manifest", "receipt"):
                draft, manifest, proof = copy.deepcopy((self.fixture.draft,self.fixture.manifest,self.proof))
                value = {"draft":draft["source"],"manifest":manifest,"receipt":proof["receipt"]}[target]
                value.pop("locator")
                with self.assertRaises(ValueError): strict_archive(self.row,draft,manifest,proof,self.fixture.commit)
        def test_storage_required(self):
            for target in ("draft", "manifest", "receipt"):
                draft, manifest, proof = copy.deepcopy((self.fixture.draft,self.fixture.manifest,self.proof))
                value={"draft":draft["source"],"manifest":manifest,"receipt":proof["receipt"]}[target]; value.pop("artifact_storage")
                with self.assertRaises(ValueError): strict_archive(self.row,draft,manifest,proof,self.fixture.commit)
        def test_all_five_members(self):
            commit=copy.deepcopy(self.fixture.commit); commit["archive_index"][self.fixture.id]["members"].pop("beats.json")
            with self.assertRaises(ValueError): strict_archive(self.row,self.fixture.draft,self.fixture.manifest,self.proof,commit)
        def test_original_npz_required(self):
            manifest=copy.deepcopy(self.fixture.manifest); manifest["npz_path"]="/tmp/wrong.npz"
            with self.assertRaises(ValueError): strict_archive(self.row,self.fixture.draft,manifest,self.proof,self.fixture.commit)
        def test_verified_member_seek_no_rehash(self):
            f=self.fixture; path=Path(self.proof["archive_verification_ref"]["path"])
            wanted={f.members[n] for n in ("draft.json","fullframe_manifest.json")}
            with patch.object(a,"read_archive",side_effect=AssertionError("rehash forbidden")):
                batch=read_verified_members(f.commit_ref,f.commit,wanted,path)
            self.assertEqual(batch["extracted"][f.members["draft.json"]],f.original_raw["draft.json"])
        def test_changed_archive_stat_rejected(self):
            f=self.fixture; archive=Path(f.commit["archive_path"]); os.utime(archive,ns=(1,1))
            with self.assertRaises(ValueError): verified_batch(f.commit_ref,f.commit,Path(self.proof["archive_verification_ref"]["path"]))
        def test_zero_original_preserved_missing_only(self):
            f=self.fixture; original=f.root/"partial"; original.mkdir()
            (original/"draft.json").write_bytes(f.original_raw["draft.json"])
            (original/"fullframe_manifest.json").write_bytes(f.original_raw["fullframe_manifest.json"])
            (original/"origin_proof.json").write_bytes(b"")
            batch=read_verified_members(f.commit_ref,f.commit,{f.members[n] for n in ("draft.json","fullframe_manifest.json")},Path(self.proof["archive_verification_ref"]["path"]))
            row,zeros=recover_row(f.id,f.commit_ref,f.commit,batch,f.root/"recovered",original)
            self.assertEqual(row["rule_draft_ref"]["path"],str(original/"draft.json"))
            self.assertEqual((original/"origin_proof.json").read_bytes(),b"")
            self.assertFalse((f.root/"recovered/sources/train"/f.id/"draft.json").exists())
            self.assertEqual(len(zeros),1)
        def test_nonzero_original_never_overwritten(self):
            f=self.fixture; original=f.root/"different"; original.mkdir();(original/"draft.json").write_bytes(b"wrong")
            batch=read_verified_members(f.commit_ref,f.commit,{f.members[n] for n in ("draft.json","fullframe_manifest.json")},Path(self.proof["archive_verification_ref"]["path"]))
            with self.assertRaises(ValueError): recover_row(f.id,f.commit_ref,f.commit,batch,f.root/"recovered",original)
            self.assertEqual((original/"draft.json").read_bytes(),b"wrong")
        def test_once_rejects_existing_even_identical(self):
            path=self.fixture.root/"evidence.json"; once(path,b"{}")
            with self.assertRaises(FileExistsError): once(path,b"{}")
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryTests))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); args=ap.parse_args()
    if args.self_test: self_test()
    else: main()
