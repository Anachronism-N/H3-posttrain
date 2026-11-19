"""Lightweight caption progress: read published indexes, never media or model files."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

BAR = Path(__file__).resolve().parent.parent
TOTAL = 222939
PRIORITY = {"failed": 0, "terminal_without_result": 0, "ready_not_joint": 1,
            "partial_candidate": 2, "complete_candidate": 3}


def source_key(row):
    sid = row.get("source_id", row.get("id"))
    split = row.get("split")
    if not isinstance(sid, str) or not sid or split not in ("train", "val", "test"):
        raise ValueError("Invalid source key in published index")
    return split + "/" + sid


class LiveCounts:
    def __init__(self, bar=BAR):
        self.bar = Path(bar)
        self.main_fingerprint = None
        self.main = {}
        self.streams = {}
        self.errors = {}
        self.main_updated = None

    def read_main(self):
        path = self.bar / "curated222939_overlay_v2/candidates.jsonl"
        if not path.exists():
            return
        before = path.stat()
        stamp = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if stamp == self.main_fingerprint:
            return
        rows = {}
        with path.open("rb") as stream:
            for line in stream:
                row = json.loads(line)
                state = row["state"]
                if state not in PRIORITY:
                    raise ValueError("Unknown candidate state")
                key = source_key(row)
                if key in rows:
                    raise ValueError("Duplicate source in main snapshot")
                rows[key] = state
        after = path.stat()
        if stamp != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            self.errors["main"] = "snapshot publication in progress; previous counts retained"
            return
        self.main, self.main_fingerprint = rows, stamp
        self.main_updated = datetime.fromtimestamp(after.st_mtime, timezone.utc).isoformat()
        self.errors.pop("main", None)

    def read_stream(self, path, ready=False):
        if not path.exists():
            return {}
        value = self.streams.setdefault(str(path), {"identity": None, "offset": 0, "rows": {}})
        st = path.stat()
        identity = (st.st_dev, st.st_ino)
        if value["identity"] is None:
            value["identity"] = identity
        if identity != value["identity"] or st.st_size < value["offset"]:
            self.errors[str(path)] = "stream replaced/truncated; cursor retained"
            return value["rows"]
        with path.open("rb") as stream:
            stream.seek(value["offset"])
            while stream.tell() < st.st_size:
                begin = stream.tell()
                line = stream.readline()
                if not line.endswith(b"\n"):
                    break
                try:
                    row = json.loads(line)
                    key = source_key(row)
                    status = row["status"]
                    if ready:
                        if status not in ("ready", "failed"):
                            raise ValueError("Unknown producer status")
                        state = "ready_not_joint" if status == "ready" else "failed"
                    else:
                        if status not in ("joint_candidate", "incomplete_candidate", "failed", "producer_failed"):
                            raise ValueError("Unknown consumer status")
                        if row.get("joint_complete") is True and status != "joint_candidate":
                            raise ValueError("Inconsistent complete status")
                        state = ("complete_candidate" if row.get("joint_complete") is True
                                 else "partial_candidate" if status == "incomplete_candidate" else "failed")
                    value["rows"][key] = state
                except (ValueError, KeyError, TypeError) as error:
                    self.errors[str(path)] = "invalid published row at %d: %s" % (begin, error)
                    break
                value["offset"] = stream.tell()
        return value["rows"]

    def poll(self):
        self.read_main()
        merged = dict(self.main)
        editions = {}
        for name, producers, consumers in (("collaborative_v3", 4, 9), ("collaborative_v4", 1, 4)):
            root = self.bar / ("curated222939_" + name)
            ready, terminal = {}, {}
            for shard in range(producers):
                ready.update(self.read_stream(root / "ready" / ("producer-%04d.jsonl" % shard), True))
            for shard in range(consumers):
                terminal.update(self.read_stream(root / "consumers" / ("shard-%04d" % shard) / "completed.jsonl"))
            editions[name] = {"prepared_ready": sum(v == "ready_not_joint" for v in ready.values()),
                              "terminal_sources": len(terminal),
                              "complete_candidates": sum(v == "complete_candidate" for v in terminal.values()),
                              "partial_candidates": sum(v == "partial_candidate" for v in terminal.values())}
            for key, state in [*ready.items(), *terminal.items()]:
                if PRIORITY[state] > PRIORITY.get(merged.get(key), -1):
                    merged[key] = state
        counts = Counter(merged.values())
        return {"read_at_UTC": datetime.now(timezone.utc).isoformat(),
                "main_snapshot_published_UTC": self.main_updated, "expected_sources": TOTAL,
                "unique_sources_seen": len(merged), "complete_candidates": counts["complete_candidate"],
                "partial_candidates": counts["partial_candidate"], "prepared_without_caption": counts["ready_not_joint"],
                "sources_not_seen": TOTAL - len(merged), "production_editions": editions,
                "index_errors": dict(self.errors), "quality_approval_inferred": False,
                "scope": "Main published candidate snapshot plus new live producer/consumer indexes; excludes separate oldbar import alternatives",
                "no_media_or_model_validation": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("positive interval required")
    state = LiveCounts()
    while True:
        value = state.poll()
        print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
