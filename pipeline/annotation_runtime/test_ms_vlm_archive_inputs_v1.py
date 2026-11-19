"""Exclusive /tmp synthetic CPU contract tests; no models or media inputs."""
import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import ms_vlm_archive_inputs_v1 as a
v = a.v
old = v.module("frozen_v2_cpu_test", a.RT / "ms_fullframe_vlm_postprocess_v2.py")


class ArchiveInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ms-archive-cpu-", dir="/tmp")
        self.root = Path(self.tmp.name)
        self.batch = self.root / "batch"
        self.batch.mkdir()
        self.id = "SyntheticDance"
        self.source = dict(id=self.id, split="train", video_path=str(self.root / (self.id + ".mp4")), source_ordinal=11)
        self.prefix = "cases/train/" + self.id
        self.members = {n: self.prefix + "/" + n for n in ("draft.json", "fullframe_manifest.json", "pose.npz", "receipt.json")}
        self.origin = str(self.batch / self.members["fullframe_manifest.json"])
        self.locator = dict(archive_path=str(self.batch / "batch.tar"), source_case_prefix=self.prefix,
                            fullframe_manifest_member=self.members["fullframe_manifest.json"], npz_member=self.members["pose.npz"])
        flags = dict(list(v.FLAGS.items())[:4])
        self.manifest = dict(schema="ms_fullframe_pose2d_manifest_v1", id=self.id, split="train",
            video_path=self.source["video_path"], video_sha256="a"*64, frame_count=3, frame_indices=[0, 1, 2],
            times=[0., .04, .08], source_timestamps=[10., 10.04, 10.08], pts_origin=10., width=1280, height=704,
            duration_seconds=.12, fullframe_manifest_path=self.origin,
            npz_path=str(self.batch / self.members["pose.npz"]), locator=self.locator, **flags)
        self.draft = dict(schema="ms_fullframe_pose2d_rule_draft_v2", id=self.id, split="train", ms_text="Visible motion 测试",
                          source=dict(fullframe_manifest_path=self.origin, video_path=self.source["video_path"],
                                      video_sha256="a"*64, locator=self.locator), **flags)
        self.rebind()
        self.receipt = dict(source_row=self.source, status=a.SUCCESS, artifact_members=self.members, locator=self.locator, **flags)
        self.item = dict(source_row=self.source, members=self.members, source_case_prefix=self.prefix)

    def tearDown(self):
        self.tmp.cleanup()

    def rebind(self):
        binding = dict(schema="ms_fullframe_pose2d_binding_v1", observations_sha256="b"*64)
        payload = dict(schema=binding["schema"], fullframe_manifest=self.manifest, observations_sha256=binding["observations_sha256"])
        binding["sha256"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.draft["source"]["fullframe_binding"] = binding

    def create(self, extras=(), mutate_member=None):
        rawdraft = json.dumps(self.draft, ensure_ascii=False, indent=3).encode() + b" \n"
        rawmanifest = json.dumps(self.manifest, ensure_ascii=False, indent=1).encode() + b"\n\n"
        rawvalues = {"draft.json": rawdraft, "fullframe_manifest.json": rawmanifest, "pose.npz": b"NPZ_OPAQUE_NOT_DECODED", "receipt.json": b"{}"}
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for n, raw in rawvalues.items():
                member = tarfile.TarInfo(self.members[n])
                member.size = len(raw)
                member.pax_headers = {"mtime": "1.125"}
                if mutate_member:
                    mutate_member(member)
                archive.addfile(member, io.BytesIO(raw) if member.isfile() else None)
            for member, raw in extras:
                archive.addfile(member, io.BytesIO(raw) if member.isfile() else None)
        raw = payload.getvalue()
        archivepath = self.batch / "batch.tar"
        archivepath.write_bytes(raw)
        self.commit = dict(schema="ms_unified_novlm_batch_commit_v3", archive_path=str(archivepath),
            archive_sha256=hashlib.sha256(raw).hexdigest(), archive_bytes=len(raw),
            archive_index={self.id: self.item}, receipts=[self.receipt])
        self.commit_ref = a.json_once(self.batch / "commit.json", self.commit)
        self.original_raw = rawvalues
        return self.commit_ref

    def bridge(self):
        return a.materialize(self.commit_ref, self.commit, [self.id], self.root / "out", {})[0]

    def rewrite_proof(self, row, change):
        proof = v.checked(row["origin_proof_ref"])
        change(proof)
        modified = dict(row)
        modified["origin_proof_ref"] = a.json_once(self.root / "changed-proof.json", proof)
        return modified

    def test_rawbytes_logical_origin_binding_and_once_only_hash(self):
        self.create()
        cache = {}
        row = a.materialize(self.commit_ref, self.commit, [self.id], self.root / "out", cache)[0]
        self.assertNotEqual(row["fullframe_origin_path"], row["fullframe_manifest_ref"]["path"])
        self.assertFalse(Path(row["fullframe_origin_path"]).exists())
        for name, key in (("draft.json", "rule_draft_ref"), ("fullframe_manifest.json", "fullframe_manifest_ref")):
            self.assertEqual(Path(row[key]["path"]).read_bytes(), self.original_raw[name])
        draft, manifest = v.input_pair(row)
        self.assertEqual(draft["source"]["fullframe_binding"], self.draft["source"]["fullframe_binding"])
        self.assertEqual(manifest["npz_path"], self.manifest["npz_path"])
        self.assertEqual(len(cache), 1)
        batch = next(iter(cache.values()))
        self.assertEqual(batch["verification"]["whole_archive_hash_reads"], 1)
        a.read_archive(self.commit_ref, self.commit, batch["wanted"], self.root / "out", cache)
        with self.assertRaises(ValueError):
            old.input_pair(row)
        self.assertEqual(a.write_raw_once(Path(row["rule_draft_ref"]["path"]), self.original_raw["draft.json"]), row["rule_draft_ref"])
        with self.assertRaises(ValueError):
            a.write_raw_once(Path(row["rule_draft_ref"]["path"]), b"changed")

    def test_wrong_row_origin(self):
        self.create(); row = self.bridge(); row["fullframe_origin_path"] += "-wrong"
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_wrong_row_id(self):
        self.create(); row = self.bridge(); row["id"] = "Wrong"
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_wrong_split(self):
        self.create(); row = self.bridge(); row["split"] = "val"
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_wrong_video(self):
        self.create(); row = self.bridge(); row["video_path"] += "-wrong"
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_wrong_proof_sha(self):
        self.create(); row = self.bridge(); row["origin_proof_ref"]["sha256"] = "0"*64
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_wrong_commit_sha(self):
        self.create(); row = self.bridge()
        row = self.rewrite_proof(row, lambda p: p["commit_ref"].update(sha256="0"*64))
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_wrong_member(self):
        self.create(); row = self.bridge()
        row = self.rewrite_proof(row, lambda p: p["members"].update({"draft.json": "cases/train/Wrong/draft.json"}))
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_wrong_member_sha(self):
        self.create(); row = self.bridge()
        row = self.rewrite_proof(row, lambda p: p["member_metadata"]["draft.json"].update(sha256="0"*64))
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_unverified_archive_hash(self):
        self.create(); row = self.bridge()
        row = self.rewrite_proof(row, lambda p: p.update(sha_verified_once=False))
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_wrong_archive_sha(self):
        self.create(); self.commit["archive_sha256"] = "0"*64
        with self.assertRaises(ValueError): self.bridge()

    def test_wrong_archive_bytes(self):
        self.create(); self.commit["archive_bytes"] += 1
        with self.assertRaises(ValueError): self.bridge()

    def test_wrong_locator(self):
        self.draft["source"]["locator"] = dict(self.locator, source_case_prefix="cases/train/Wrong")
        self.create()
        with self.assertRaises(ValueError): self.bridge()

    def test_wrong_original_manifest_origin(self):
        self.manifest["fullframe_manifest_path"] += "-wrong"; self.rebind(); self.create()
        with self.assertRaises(ValueError): self.bridge()

    def test_wrong_original_binding(self):
        self.draft["source"]["fullframe_binding"]["sha256"] = "0"*64; self.create()
        with self.assertRaises(ValueError): self.bridge()

    def test_failed_and_oldclaim_rejected(self):
        for status in ("failed_sealed_no_retry", "existing_claim_not_reassigned"):
            with self.subTest(status=status), self.assertRaises(ValueError):
                a.eligible(self.source, dict(self.receipt, status=status), set(), set())

    def test_valtest_cannot_be_train(self):
        for split in ("val", "test"):
            source = dict(self.source, split=split)
            with self.subTest(split=split), self.assertRaises(ValueError):
                a.eligible(source, dict(self.receipt, source_row=source), set(), set())

    def test_old500_exclusion_and_casefold_duplicates(self):
        with self.assertRaises(ValueError): a.eligible(self.source, self.receipt, {self.id.casefold()}, set())
        with self.assertRaises(ValueError): a.eligible(self.source, self.receipt, set(), {self.id.casefold()})

    def test_nonregular_link_rejected(self):
        member = tarfile.TarInfo("extra-link"); member.type = tarfile.SYMTYPE; member.linkname = "/tmp/untrusted"
        self.create([(member, b"")])
        with self.assertRaises(ValueError): self.bridge()

    def test_pax_sparse_rejected(self):
        member = tarfile.TarInfo("sparse"); member.pax_headers = {"GNU.sparse.major": "1", "GNU.sparse.minor": "0"}
        with self.assertRaises(ValueError): a.safe_member(member)
        member.pax_headers = {"SCHILY.realsize": "1"}
        with self.assertRaises(ValueError): a.safe_member(member)

    def test_pax_sparse_tar_rejected_before_payload_decode(self):
        member = tarfile.TarInfo("sparse.npz"); member.size = 1
        member.pax_headers = {"GNU.sparse.map": "0,1", "GNU.sparse.size": "1"}
        self.create([(member, b"x")])
        with self.assertRaises(ValueError): self.bridge()

    def test_duplicate_member_rejected(self):
        member = tarfile.TarInfo(self.members["draft.json"]); member.size = 2
        self.create([(member, b"{}")])
        with self.assertRaises(ValueError): self.bridge()

    def test_traversal_rejected(self):
        member = tarfile.TarInfo("../escape.json"); member.size = 2
        self.create([(member, b"{}")])
        with self.assertRaises(ValueError): self.bridge()

    def test_json_member_size_bound(self):
        self.draft["padding"] = "x" * (a.MAX_JSON_BYTES + 1); self.create()
        with self.assertRaises(ValueError): self.bridge()

    def test_archive_size_bound(self):
        self.create()
        with patch.object(a, "MAX_ARCHIVE_BYTES", 100), self.assertRaises(ValueError): self.bridge()

    def test_old_v2_contract_preserved(self):
        location = self.root / "old-manifest.json"
        self.manifest["fullframe_manifest_path"] = str(location)
        self.draft["source"]["fullframe_manifest_path"] = str(location)
        self.rebind()
        row = dict(id=self.id, split="train", video_path=self.source["video_path"],
            rule_draft_ref=a.json_once(self.root / "old-draft.json", self.draft),
            fullframe_manifest_ref=a.json_once(location, self.manifest))
        self.assertEqual(v.input_pair(row), old.input_pair(row))

    def test_origin_fields_must_be_paired(self):
        self.create(); row = self.bridge(); del row["origin_proof_ref"]
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_original_lifecycle_160gib_and_model_recipe_unchanged(self):
        trees = [ast.parse((a.RT / name).read_text()) for name in ("ms_fullframe_vlm_postprocess_v2.py", "ms_fullframe_vlm_postprocess_v3.py")]
        unchanged = ("gate", "runtime", "call_stage", "run_sources", "validate", "previous_safe", "worker", "main", "actual_model", "request_for", "evidence_policy_violations")
        for name in unchanged:
            nodes = [next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == name) for t in trees]
            self.assertEqual(ast.dump(nodes[0]), ast.dump(nodes[1]), name)
        self.assertEqual(v.base.required_budget(dict(gpus=[2, 3])), 160 * 2**30)
        self.assertEqual(v.FLAGS, old.FLAGS)
        self.assertEqual(v.HOSTS, old.HOSTS)
        self.assertEqual(v.PAIRS, old.PAIRS)
        self.assertEqual(v.SELF, a.RT / "ms_fullframe_vlm_postprocess_v3.py")
        self.assertEqual(v.SCHEMA, "ms_fullframe_vlm_postprocess_native_v3")
        self.assertEqual(v.PREFIX, "ms_fullframe_vlm_postprocess_v3_")


if __name__ == "__main__":
    unittest.main(verbosity=2)
