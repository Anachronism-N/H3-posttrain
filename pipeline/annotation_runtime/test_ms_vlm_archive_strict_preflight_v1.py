"""CPU-only synthetic tests for a separate immutable strict preflight layer."""
import json
import tarfile
import unittest

import ms_vlm_archive_strict_preflight_v1 as s
import test_ms_vlm_archive_inputs_v1 as original


class StrictPreflightTests(unittest.TestCase):
    def setUp(self):
        self.fixture = original.ArchiveInputTests(methodName="test_rawbytes_logical_origin_binding_and_once_only_hash")
        f = self.fixture
        f.setUp()
        f.members["beats.json"] = f.prefix + "/beats.json"
        for value in (f.draft["source"], f.manifest, f.receipt):
            value.update(artifact_storage="archive_member", artifact_members=f.members)
        f.draft.update(locator=f.locator, artifact_storage="archive_member")
        f.draft["source"]["npz_path"] = f.manifest["npz_path"]
        f.receipt.update(schema="ms_unified_novlm_case_receipt_v2", vlm_called=False)
        f.item["receipt_member"] = f.members["receipt.json"]

    def tearDown(self):
        self.fixture.tearDown()

    def bridge(self):
        f = self.fixture
        f.rebind()
        member = tarfile.TarInfo(f.members["beats.json"])
        member.size = 2
        f.create([(member, b"{}")])
        f.commit.update(batch_sha256=f.commit["archive_sha256"], vlm_called=False)
        f.commit_ref = s.a.json_once(f.batch / "strict-commit.json", f.commit)
        return f.bridge()

    def rejects(self):
        with self.assertRaises(ValueError):
            s.strict_row(self.bridge())

    def test_complete_actual_shape_passes_top_members_absent(self):
        row = self.bridge()
        self.assertEqual(len(s.strict_row(row)), 5)
        self.assertNotIn("artifact_members", self.fixture.draft)

    def test_missing_source_locator_rejected(self):
        del self.fixture.draft["source"]["locator"]
        self.rejects()

    def test_missing_manifest_locator_rejected(self):
        del self.fixture.manifest["locator"]
        self.rejects()

    def test_missing_receipt_locator_rejected(self):
        del self.fixture.receipt["locator"]
        self.rejects()

    def test_missing_top_locator_rejected(self):
        del self.fixture.draft["locator"]
        self.rejects()

    def test_wrong_source_storage_rejected(self):
        self.fixture.draft["source"]["artifact_storage"] = "per_source"
        self.rejects()

    def test_wrong_manifest_storage_rejected(self):
        self.fixture.manifest["artifact_storage"] = "per_source"
        self.rejects()

    def test_wrong_receipt_storage_rejected(self):
        self.fixture.receipt["artifact_storage"] = "per_source"
        self.rejects()

    def test_wrong_top_storage_rejected(self):
        self.fixture.draft["artifact_storage"] = "per_source"
        self.rejects()

    def test_missing_source_membermap_rejected(self):
        del self.fixture.draft["source"]["artifact_members"]
        self.rejects()

    def test_missing_manifest_membermap_rejected(self):
        del self.fixture.manifest["artifact_members"]
        self.rejects()

    def test_wrong_top_members_if_present_rejected(self):
        self.fixture.draft["artifact_members"] = {"draft.json": "wrong"}
        self.rejects()

    def test_wrong_logical_npz_rejected(self):
        self.fixture.draft["source"]["npz_path"] += "-wrong"
        self.rejects()

    def test_wrong_receipt_schema_rejected(self):
        self.fixture.receipt["schema"] = "wrong"
        self.rejects()

    def test_receipt_vlm_called_rejected(self):
        self.fixture.receipt["vlm_called"] = True
        self.rejects()

    def test_wrong_receipt_member_rejected(self):
        self.fixture.item["receipt_member"] = "wrong"
        self.rejects()

    def test_exact_5member_set_rejects_sixth(self):
        f = self.fixture
        f.members["extra.json"] = f.prefix + "/extra.json"
        with self.assertRaises(ValueError):
            s.strict_row(self.bridge())


if __name__ == "__main__":
    unittest.main(verbosity=2)
