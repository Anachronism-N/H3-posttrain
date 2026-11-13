"""Offline audit of stored music-judgment fields; never reruns or edits a judge.

The proposed score follows the declared conflict-first rubric. This does not
authenticate historical runner code, raw model responses, or musical quality.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from statistics import mean

from .io import loads, write_json


STATUSES = {"valid", "unknown", "invalid", "failed", "generation_not_evaluable", "not_attempted"}
LIMITATION = (
    "Stored judgment-field audit only. Verify the historical frozen code SHA and "
    "original raw responses before adopting any proposed scores. No musical quality, "
    "human agreement, or change in expert ranking is established. Means are conditional "
    "on the separately reported valid counts; missing/failed slots receive no score. "
    "Grouping uses the stored method_id, which may be a per-case blind alias, not a "
    "generator identity. Bind the private mapping and generation-status inventory "
    "before any model-level aggregation; this audit does not perform that unblinding."
)


def proposed_score(rating):
    """Apply clear -> uncertain -> correspondence, with no inference from prose."""
    if not isinstance(rating, dict):
        raise ValueError("Rating must be an object")
    if rating.get("conflict") not in {"none", "clear", "uncertain"}:
        raise ValueError("Invalid conflict field")
    if rating.get("correspondence") not in {"absent", "partial", "explicit"}:
        raise ValueError("Invalid correspondence field")
    if type(rating.get("full_support")) is not bool:
        raise ValueError("full_support must be boolean")
    if rating["conflict"] == "clear":
        return 1
    if rating["conflict"] == "uncertain":
        return None
    if rating["correspondence"] == "absent":
        return 2
    if rating["correspondence"] == "partial":
        return 3
    return 5 if rating["full_support"] else 4


def _audit_row(row, line):
    if not isinstance(row, dict):
        raise ValueError("Line %d: expected an object" % line)
    item_id = row.get("item_sha256")
    if not isinstance(item_id, str) or not re.fullmatch(r"[0-9a-f]{64}", item_id):
        raise ValueError("Line %d: invalid item_sha256" % line)
    method = row.get("method_id")
    if not isinstance(method, str) or not method.strip() or len(method) > 256 or any(ord(c) < 32 for c in method):
        raise ValueError("Line %d: invalid method_id" % line)
    if row.get("status") not in STATUSES:
        raise ValueError("Line %d: unsupported source status" % line)
    for name in ("case_id", "evidence_id"):
        if name in row and (not isinstance(row[name], str) or not row[name].strip()):
            raise ValueError("Line %d: invalid %s" % (line, name))
    result = dict(source_line=line, item_sha256=item_id, method_id=method,
                  source_status=row["status"], source_row=row,
                  audit_status="retained_unscored_status", eligible_for_recompute=False,
                  original_score=None, proposed_score=None, changed=False,
                  uncertain_mis_scored=False, retained_in_registered_denominator=True)
    if row["status"] != "valid":
        return result
    rating = row.get("rating")
    if rating is None:
        result["audit_status"] = "missing_rating"
        return result
    if not isinstance(rating, dict) or "music_fit" not in rating:
        raise ValueError("Line %d: valid rating requires recorded music_fit" % line)
    old = rating["music_fit"]
    if old is not None and (type(old) is not int or not 1 <= old <= 5):
        raise ValueError("Line %d: original music_fit must be null or integer 1..5" % line)
    try:
        new = proposed_score(rating)
    except (ValueError, TypeError) as error:
        raise ValueError("Line %d: %s" % (line, error)) from error
    result.update(audit_status="recomputed_from_stored_fields", eligible_for_recompute=True,
                  original_score=old, proposed_score=new, changed=old != new,
                  uncertain_mis_scored=rating["conflict"] == "uncertain" and old is not None)
    return result


def audit_ratings(ratings_path, out_dir):
    source, output = Path(ratings_path).resolve(), Path(out_dir).resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite audit directory: " + str(output))
    data = source.read_bytes()
    source_sha = hashlib.sha256(data).hexdigest()
    records, seen = [], set()
    for number, text in enumerate(data.decode("utf-8").splitlines(), 1):
        if not text.strip():
            continue
        try:
            record = _audit_row(loads(text), number)
        except (ValueError, TypeError) as error:
            raise ValueError("Invalid ratings JSONL at line %d: %s" % (number, error)) from error
        if record["item_sha256"] in seen:
            raise ValueError("Duplicate item_sha256 at line %d" % number)
        seen.add(record["item_sha256"])
        records.append(record)
    if not records:
        raise ValueError("Ratings file has no registered rows")
    groups = defaultdict(list)
    for record in records:
        groups[record["method_id"]].append(record)
    methods = {}
    for method, rows in sorted(groups.items()):
        old = [r["original_score"] for r in rows if r["original_score"] is not None]
        new = [r["proposed_score"] for r in rows if r["proposed_score"] is not None]
        methods[method] = dict(
            registered=len(rows), source_status_counts=dict(Counter(r["source_status"] for r in rows)),
            recomputable_rows=sum(r["eligible_for_recompute"] for r in rows),
            missing_rating_rows=sum(r["audit_status"] == "missing_rating" for r in rows),
            original_valid_count=len(old), proposed_valid_count=len(new),
            original_unscored_count=len(rows) - len(old), proposed_unscored_count=len(rows) - len(new),
            uncertain_mis_scored_count=sum(r["uncertain_mis_scored"] for r in rows),
            changed_score_count=sum(r["changed"] for r in rows),
            original_mean=mean(old) if old else None, proposed_mean=mean(new) if new else None)
    summary = dict(
        schema="captionlab.music_score_audit.v1", source_path=str(source),
        source_sha256=source_sha, source_bytes=len(data), tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        registered=len(records), methods=methods,
        proposed_rule="clear=1; uncertain=null; none+absent=2; none+partial=3; none+explicit=4 or 5 with full_support",
        hypothetical_rescoring=True, historical_code_verified=False, raw_model_responses_verified=False,
        generator_identities_verified=False, generation_inventory_verified=False,
        grouping_scope="stored method_id only; potentially per-case randomized blind alias",
        quality_validated=False, expert_selection_change_established=False,
        original_scores_modified=False, model_calls=0, limitation=LIMITATION)
    if source.read_bytes() != data:
        raise ValueError("Source ratings changed during audit")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "row_audit.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    summary["row_audit_sha256"] = hashlib.sha256((output / "row_audit.jsonl").read_bytes()).hexdigest()
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    summary = audit_ratings(args.ratings, args.out_dir)
    print(json.dumps(dict(registered=summary["registered"], methods=summary["methods"],
                          hypothetical_rescoring=True, model_calls=0), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
