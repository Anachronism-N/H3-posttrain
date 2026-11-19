import json
from pathlib import Path
import tempfile
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import caption_live_counts as m


class CountsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.state = m.LiveCounts(self.root)

    def save(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"".join((json.dumps(row) + "\n").encode() for row in rows))

    def test_main_and_live_complete_are_source_deduplicated(self):
        self.save(self.root / "curated222939_overlay_v2/candidates.jsonl",
                  [{"source_id": "one", "split": "train", "state": "partial_candidate"}])
        p = self.root / "curated222939_collaborative_v3/consumers/shard-0000/completed.jsonl"
        row = {"id": "one", "split": "train", "status": "joint_candidate", "joint_complete": True}
        self.save(p, [row, row])
        v = self.state.poll()
        self.assertEqual(v["unique_sources_seen"], 1)
        self.assertEqual(v["complete_candidates"], 1)
        self.assertEqual(v["partial_candidates"], 0)
        self.assertFalse(v["quality_approval_inferred"])

    def test_incremental_append_and_partial_tail(self):
        p = self.root / "curated222939_collaborative_v4/ready/producer-0000.jsonl"
        row = {"id": "one", "split": "train", "status": "ready"}
        self.save(p, [row])
        self.assertEqual(self.state.poll()["prepared_without_caption"], 1)
        with p.open("ab") as s:
            s.write(b'{"id":"two",')
        self.assertEqual(self.state.poll()["unique_sources_seen"], 1)
        with p.open("ab") as s:
            s.write(b'"split":"train","status":"ready"}\n')
        self.assertEqual(self.state.poll()["unique_sources_seen"], 2)

    def test_unknown_row_retains_cursor_and_does_not_skip(self):
        p = self.root / "curated222939_collaborative_v3/ready/producer-0000.jsonl"
        p.parent.mkdir(parents=True)
        p.write_bytes(b'{"id":"one","split":"train","status":"other"}\n')
        v = self.state.poll()
        self.assertTrue(v["index_errors"])
        self.assertEqual(self.state.streams[str(p)]["offset"], 0)
        self.assertEqual(v["unique_sources_seen"], 0)

    def test_replaced_stream_is_not_reinterpreted(self):
        p = self.root / "curated222939_collaborative_v3/ready/producer-0000.jsonl"
        self.save(p, [{"id": "one", "split": "train", "status": "ready"}])
        self.state.poll()
        self.save(p, [])
        v = self.state.poll()
        self.assertTrue(v["index_errors"])
        self.assertEqual(v["unique_sources_seen"], 1)

    def test_main_snapshot_only_reloads_on_publication(self):
        p = self.root / "curated222939_overlay_v2/candidates.jsonl"
        self.save(p, [{"source_id": "one", "split": "val", "state": "ready_not_joint"}])
        self.state.poll()
        original = self.state.main
        self.state.poll()
        self.assertIs(self.state.main, original)
        self.save(p, [{"source_id": "one", "split": "val", "state": "complete_candidate"}])
        self.assertEqual(self.state.poll()["complete_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
