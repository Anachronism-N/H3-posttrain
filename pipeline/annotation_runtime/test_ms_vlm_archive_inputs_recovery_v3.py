"""Independent synthetic CPU tests for missing and durability closure."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import ms_vlm_archive_inputs_recovery_v3 as x

class RecoveryV3Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir="/tmp",prefix="archive-v3-cpu-")
        self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_futurepending_missing_config_explicit_plan(self):
        row=dict(id="Fresh",split="train");p=self.root/"plan.jsonl";raw=x.a.encoded(row);p.write_bytes(raw)
        job=dict(config_path=str(self.root/"future.json"),expected_running=False,worker_status="future_pending",
                 sources=1,plan_ref=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest()))
        rows,ref,cfg=x.job_plan(job);self.assertEqual(rows,[row]);self.assertIsNone(cfg)
    def test_ordinary_missing_config_rejected(self):
        with self.assertRaises(ValueError):x.job_plan(dict(config_path=str(self.root/"no.json"),expected_running=False))
    def test_running_futurepending_missing_rejected(self):
        with self.assertRaises(ValueError):x.job_plan(dict(config_path=str(self.root/"no.json"),expected_running=True,worker_status="future_pending",plan_ref={}))
    def test_wrong_futurepending_sha_rejected(self):
        p=self.root/"plan.jsonl";p.write_bytes(x.a.encoded(dict(id="Fresh",split="train")))
        with self.assertRaises(ValueError):x.job_plan(dict(config_path=str(self.root/"no.json"),expected_running=False,
            worker_status="pending_config",sources=1,plan_ref=dict(path=str(p),sha256="0"*64)))
    def test_durability_existing_exact_bytes_no_write(self):
        p=self.root/"draft.json";p.write_bytes(b"{ }\n");before=p.stat()
        ref=x.sync_existing(p,b"{ }\n");self.assertEqual(p.read_bytes(),b"{ }\n")
        self.assertEqual(x.a.signature(before),x.a.signature(p.stat()));self.assertEqual(ref["sha256"],hashlib.sha256(b"{ }\n").hexdigest())
    def test_durability_mismatch_no_overwrite(self):
        p=self.root/"draft.json";p.write_bytes(b"old")
        with self.assertRaises(ValueError):x.sync_existing(p,b"wrong")
        self.assertEqual(p.read_bytes(),b"old")
    def test_durability_zero_original_not_rewritten(self):
        p=self.root/"proof.json";p.write_bytes(b"")
        with self.assertRaises(ValueError):x.sync_existing(p,b"{ }")
        self.assertEqual(p.read_bytes(),b"")
    def test_new_once_rejects_identical_original(self):
        p=self.root/"new.json";x.old.once(p,b"{}")
        with self.assertRaises(FileExistsError):x.old.once(p,b"{}")

class MissingSourceTests(unittest.TestCase):
    def setUp(self):
        tests=x.v.module("v3_archive_fixtures",x.RT/"test_ms_vlm_archive_inputs_v1.py")
        self.f=tests.ArchiveInputTests("test_rawbytes_logical_origin_binding_and_once_only_hash");self.f.setUp();f=self.f
        f.members["beats.json"]=f.prefix+"/beats.json";f.item["receipt_member"]=f.members["receipt.json"]
        for obj in (f.draft,f.draft["source"],f.manifest,f.receipt):obj["artifact_storage"]="archive_member";obj["locator"]=f.locator
        f.manifest["artifact_members"]=f.members;f.draft["source"]["artifact_members"]=f.members
        f.draft["source"]["npz_path"]=f.manifest["npz_path"];f.rebind()
        m=x.old.tarfile.TarInfo(f.members["beats.json"]);m.size=2;f.create(extras=[(m,b"{}")]);self.row=f.bridge()
        proof=x.v.checked(self.row["origin_proof_ref"])
        self.batch=x.old.read_verified_members(f.commit_ref,f.commit,{f.members[n]for n in ("draft.json","fullframe_manifest.json")},Path(proof["archive_verification_ref"]["path"]))
    def tearDown(self):self.f.tearDown()
    def test_missing_manifest_and_zero_proof_preserve_draft(self):
        f=self.f;old=f.root/"partial";old.mkdir();(old/"draft.json").write_bytes(f.original_raw["draft.json"]);(old/"origin_proof.json").write_bytes(b"")
        with patch.object(x.a,"read_archive",side_effect=AssertionError("Existing proof must not rehash")):
            row,adopted,zeros=x.restore_row(f.id,f.commit_ref,f.commit,self.batch,f.root/"v3",old)
        self.assertEqual(row["rule_draft_ref"]["path"],str(old/"draft.json"));self.assertEqual(len(adopted),1);self.assertEqual(len(zeros),1)
        self.assertFalse((f.root/"v3/sources/train"/f.id/"draft.json").exists());self.assertEqual((old/"origin_proof.json").read_bytes(),b"")
    def test_complete_partial_proof_reused_and_synced(self):
        f=self.f;old=Path(self.row["origin_proof_ref"]["path"]).parent
        row,adopted,zeros=x.restore_row(f.id,f.commit_ref,f.commit,self.batch,f.root/"v3",old)
        self.assertEqual(row,self.row);self.assertEqual(len(adopted),3);self.assertFalse(zeros)
    def test_extra_draft_top_locator_storage_and_manifest_members(self):
        f=self.f;proof=x.v.checked(self.row["origin_proof_ref"])
        for target,key in (("draft","locator"),("draft","artifact_storage"),("manifest","artifact_members")):
            d,m=copy.deepcopy((f.draft,f.manifest));{"draft":d,"manifest":m}[target].pop(key)
            with self.assertRaises(ValueError):x.strict_extra(self.row,d,m,proof,f.commit)

if __name__=="__main__":unittest.main(verbosity=2)
