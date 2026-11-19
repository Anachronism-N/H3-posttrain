"""CPU-only snapshot boundary and hot-path tests; fixtures never touch source data."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import curated_source_snapshot_reader_v1 as m


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="source-snapshot-tests-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = self.root / "original.json"
        self.original.write_bytes(b'{"safe":[1,2]}\n')
        raw, identity = m.pilot.stable_bytes(self.original)
        source = {"job": {"id": "fixture", "source_key": "fixture/source"},
                  "refs": {str(self.original): (m.pilot.digest(raw), identity, str(self.original.resolve()))}}
        self.entries, _ = m.pilot.build_bundle(source, self.root)
        self.reader = m.SourceSnapshotReader("fixture/source", self.entries)
        self.raw = raw

    def test_repeated_reads_have_no_filesystem_or_content_hash_calls(self):
        with self.reader:
            with patch.object(Path, "resolve", side_effect=AssertionError("hot resolve")), \
                 patch.object(m.pilot, "stable_bytes", side_effect=AssertionError("hot file read")), \
                 patch.object(m.pilot.os, "stat", side_effect=AssertionError("hot stat")), \
                 patch.object(m.pilot, "digest", side_effect=AssertionError("hot SHA")):
                for _ in range(100):
                    self.assertEqual(self.reader.read(self.original), self.raw)
                    self.assertEqual(self.reader.sha(self.original), self.entries[str(self.original)].expected_sha256)
                    self.assertEqual(self.reader.hash(self.raw), self.entries[str(self.original)].expected_sha256)
        self.assertTrue(self.reader.committed)
        self.assertEqual(self.reader.boundary_stats["files_checked"], 4)
        self.assertEqual(self.reader.stats["explicit_stat_calls"], 0)
        self.assertEqual(self.reader.stats["explicit_resolve_calls"], 0)

    def test_changed_original_before_entry_rejected(self):
        self.original.write_bytes(b'{"safe":[9,2]}\n')
        with self.assertRaisesRegex(ValueError, "Original content changed"):
            with self.reader:
                self.fail("entered changed source")
        self.assertFalse(self.reader.committed)

    def test_changed_local_even_with_refreshed_stat_rejected(self):
        entry = self.entries[str(self.original)]
        local = Path(entry.local_path)
        local.chmod(0o600)
        local.write_bytes(b'{"safe":[9,2]}\n')
        self.entries[str(self.original)] = replace(entry, local_stat=m.pilot.stable_bytes(local)[1])
        reader = m.SourceSnapshotReader("fixture/source", self.entries)
        with self.assertRaisesRegex(ValueError, "Local snapshot content changed"):
            with reader:
                pass

    def test_mid_source_change_keeps_snapshot_but_rejects_commit(self):
        with self.assertRaisesRegex(ValueError, "Original content changed"):
            with self.reader:
                self.original.write_bytes(b'{"safe":[9,2]}\n')
                self.assertEqual(self.reader.read(self.original), self.raw)
        self.assertFalse(self.reader.committed)
        self.assertFalse(self.reader._active)
        self.assertFalse(self.reader._bytes)

    def test_mid_local_change_rejects_commit(self):
        with self.assertRaisesRegex(ValueError, "Local snapshot content changed"):
            with self.reader:
                local = Path(self.entries[str(self.original)].local_path)
                local.chmod(0o600)
                local.write_bytes(b'{"safe":[9,2]}\n')
                self.assertEqual(self.reader.read(self.original), self.raw)
        self.assertFalse(self.reader.committed)

    def test_source_mapping_and_logical_path_mismatch_rejected(self):
        for change in ({"source_key": "fixture/other"}, {"logical_path": str(self.root / "other.json")}):
            entries = dict(self.entries)
            entries[str(self.original)] = replace(entries[str(self.original)], **change)
            with self.assertRaisesRegex(ValueError, "Source closure identity"):
                with m.SourceSnapshotReader("fixture/source", entries):
                    pass

    def test_unknown_path_and_wrong_expected_hash_rejected(self):
        with self.reader:
            with self.assertRaisesRegex(ValueError, "Path outside source closure"):
                self.reader.read(self.root / "missing.json")
            with self.assertRaisesRegex(ValueError, "Expected SHA mismatch"):
                self.reader.checked({"path": str(self.original), "sha256": "0" * 64})

    def test_json_parse_once_returns_independent_mutable_values(self):
        with self.reader, patch.object(m.json, "loads", wraps=m.json.loads) as loads:
            first = self.reader.load(self.original)
            first["safe"].append(9)
            self.assertEqual(self.reader.load(self.original), {"safe": [1, 2]})
            self.assertEqual(self.reader.checked({"path": str(self.original),
                             "sha256": self.entries[str(self.original)].expected_sha256}), {"safe": [1, 2]})
            self.assertEqual(loads.call_count, 1)

    def test_exception_does_not_commit_and_clears_snapshot(self):
        with self.assertRaisesRegex(RuntimeError, "consumer failed"):
            with self.reader:
                raise RuntimeError("consumer failed")
        self.assertFalse(self.reader.committed)
        self.assertFalse(self.reader._bytes)
        with self.assertRaisesRegex(ValueError, "outside source context"):
            self.reader.read(self.original)

    def test_context_reuse_and_unbound_hash_denied(self):
        with self.assertRaisesRegex(ValueError, "outside source context"):
            self.reader.hash(self.raw)
        with self.reader:
            self.assertEqual(self.reader.hash(b"new content"), m.pilot.digest(b"new content"))
        with self.assertRaisesRegex(ValueError, "reentry"):
            with self.reader:
                pass

    def test_symlink_target_change_denied_at_exit(self):
        other = self.root / "other.json"
        other.write_bytes(self.raw)
        link = self.root / "source-link.json"
        link.symlink_to(self.original)
        raw, identity = m.pilot.stable_bytes(link)
        source = {"job": {"id": "links", "source_key": "fixture/source"},
                  "refs": {str(link): (m.pilot.digest(raw), identity, str(link.resolve()))}}
        entries, _ = m.pilot.build_bundle(source, self.root)
        reader = m.SourceSnapshotReader("fixture/source", entries)
        with self.assertRaisesRegex(ValueError, "Original path identity changed"):
            with reader:
                link.unlink()
                link.symlink_to(other)
                self.assertEqual(reader.read(link), raw)
        self.assertFalse(reader.committed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
