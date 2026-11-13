"""Offline, source-bound blinded review packs for raw caption baselines.

Only public/ may be shared for quality review. private/ contains method mappings
and optional training references. A separate reference phase lives entirely in
private/reference/ and is not a second quality evaluation. No providers run.
"""

import argparse
import copy
import hashlib
import html
import io
import json
import math
from pathlib import Path
import random
import secrets
import wave

from choreobench.baseline import _sealed, _verify
from choreobench.io import object_sha, read_json_snapshot, write_json
from .choreo_quality import (
    DIMENSIONS, QualityError, make_review_item, review_template, rubric,
    summarize_quality, validate_review,
)


class BundleError(ValueError):
    """Invalid source inventory, blind binding or returned review."""


def _require(condition, message):
    if not condition:
        raise BundleError(message)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _text(value, label):
    _require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty text")
    return value


def _bytes_sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _snapshot(path, sources):
    path = Path(path).expanduser().resolve(strict=True)
    value, digest = read_json_snapshot(path)
    _require(str(path) not in sources or sources[str(path)] == digest, "Source changed during collection")
    sources[str(path)] = digest
    return value


def _rows_bytes(rows):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows).encode("utf-8")


def _clock(row):
    audio = row.get("audio")
    _require(isinstance(audio, dict) and set(audio) == {"path", "sha256", "sample_rate", "num_samples"}, "Require explicit canonical audio fields")
    _text(audio["path"], "audio path")
    _require(_sha(audio["sha256"]) and row.get("audio_sha256") == audio["sha256"], "Row audio SHA mismatch")
    rate, count, duration = audio["sample_rate"], audio["num_samples"], row.get("duration_seconds")
    _require(type(rate) is int and 1 <= rate <= 192000 and type(count) is int and count > 0, "Invalid audio clock")
    _require(type(duration) in (int, float), "duration_seconds must be finite numeric, not bool")
    try:
        valid = math.isfinite(duration) and duration > 0 and duration == count / rate
    except (ValueError, OverflowError):
        valid = False
    _require(valid, "Audio duration differs from the integer clock")
    return audio["sha256"], rate, count, duration


def _load_result(path, sources):
    path = Path(path).expanduser().resolve(strict=True)
    result = _verify(_snapshot(path, sources))
    contract = _verify(_snapshot(path.parent / "contract.json", sources))
    _require(result.get("schema") == "captionlab.baseline_results.v1" and result.get("complete") is True,
             "Require a published baseline results.json, including its failed/unknown rows")
    _require(contract.get("schema") == "captionlab.baseline_contract.v1" and result.get("contract_sha256") == object_sha(contract),
             "Baseline contract binding mismatch")
    method = _text(result.get("method_id"), "method_id")
    _require(contract.get("method_id") == method, "Contract/result method mismatch")
    _require(type(result.get("synthetic")) is bool and result["synthetic"] is contract.get("synthetic"), "Synthetic declaration mismatch")
    _require(result.get("quality_verified") is False, "Baseline results must not assert verified quality")
    cases = contract.get("cases")
    _require(isinstance(cases, list) and bool(cases), "Baseline contract has no registered cases")
    expected = {}
    for case in cases:
        _require(isinstance(case, dict), "Malformed registered case")
        cid = _text(case.get("case_id"), "case_id")
        _require(cid not in expected, "Duplicate registered case")
        expected[cid] = case
    rows = result.get("rows")
    _require(isinstance(rows, list) and bool(rows) and type(result.get("case_count")) is int and result["case_count"] == len(rows), "Result case count mismatch")
    indexed = {}
    for record in rows:
        row = _verify(record)
        cid = _text(row.get("case_id"), "case_id")
        _require(cid in expected and cid not in indexed, "Duplicate or unregistered baseline row")
        case = expected[cid]
        _require(row.get("method_id") == method and row.get("call_id") == method + "/" + cid, "Row method/call binding mismatch")
        _require(row.get("status") in ("complete", "failed", "unknown"), "Unknown baseline execution status")
        _require(isinstance(row.get("text"), str), "Raw generation text must be retained as text")
        _text(row.get("source_group_id"), "source_group_id")
        _text(row.get("split"), "split")
        for field in ("source_group_id", "split", "duration_seconds", "audio"):
            _require(object_sha(row.get(field)) == object_sha(case.get(field)), "Row/contract " + field + " differs")
        _clock(row)
        _require(_sha(row.get("started_record_sha256")) and _sha(row.get("raw_record_sha256")), "Missing source call bindings")
        indexed[cid] = row
    _require(set(indexed) == set(expected), "Baseline rows omit registered cases")
    for status, field in (("complete", "complete_count"), ("failed", "failed_count"), ("unknown", "unknown_count")):
        _require(type(result.get(field)) is int and result[field] == sum(r["status"] == status for r in rows), "Baseline status count mismatch")
    _require(type(result.get("calls_used")) is int and result["calls_used"] == len(rows), "Baseline call count mismatch")
    raw_records = _rows_bytes(rows)
    _require(result.get("records_sha256") == _bytes_sha(raw_records), "Baseline records SHA mismatch")
    records_path = path.parent / "records.jsonl"
    actual = records_path.read_bytes()
    _require(actual == raw_records, "Published records.jsonl differs from result rows")
    sources[str(records_path.resolve())] = _bytes_sha(actual)
    manifest = contract.get("manifest_snapshot")
    _require(isinstance(manifest, dict) and isinstance(manifest.get("selection"), dict), "Missing frozen selection")
    return method, indexed, contract, result


def _reference_map(path, cases, contracts, sources):
    if path is None:
        return {}
    reference = _snapshot(path, sources)
    _require(isinstance(reference, dict) and reference.get("schema") == "captionlab.sol61_references.v1"
             and isinstance(reference.get("records"), list), "Expected sol61 private references")
    _require(isinstance(reference.get("source_files"), dict) and bool(reference["source_files"]), "Reference source-file bindings are missing")
    for contract in contracts:
        manifest = contract["manifest_snapshot"]
        _require(reference.get("selection_sha256") == object_sha(manifest["selection"]), "Reference selection mismatch")
        _require(reference.get("source_files") == manifest.get("dataset", {}).get("source_files"), "Reference source-file binding mismatch")
    refs = {}
    for row in reference["records"]:
        _require(isinstance(row, dict), "Malformed reference row")
        cid = row.get("case_id")
        _require(isinstance(cid, str) and cid in cases and cid not in refs, "Duplicate or unexpected reference case")
        _require(row.get("source_group_id") == cases[cid]["source_group_id"] and row.get("split") == cases[cid]["split"], "Reference source/split mismatch")
        raw, action = row.get("raw_caption"), row.get("action_reference")
        _require(isinstance(raw, str) and isinstance(action, str) and bool(action.strip()), "Reference action body is missing")
        _require(raw.partition("\n")[2] == action and row.get("raw_caption_sha256") == _bytes_sha(raw.encode("utf-8")), "Reference text SHA/body mismatch")
        _require(not any(line.strip().casefold().startswith("dance style:") for line in action.splitlines()), "Reference action body must exclude style headers")
        refs[cid] = copy.deepcopy(row)
    _require(set(refs) == set(cases), "References omit registered cases")
    return refs


def _audio_bytes(row, result_dir, cache):
    audio = row["audio"]
    path = Path(audio["path"]).expanduser()
    path = path.resolve(strict=True) if path.is_absolute() else (result_dir / path).resolve(strict=True)
    binding = _clock(row)
    key = str(path)
    if key in cache:
        _require(cache[key][0] == binding, "One audio path has inconsistent declared identities")
        return cache[key][1]
    raw = path.read_bytes()
    _require(_bytes_sha(raw) == binding[0], "Actual audio SHA differs from the baseline row")
    with wave.open(io.BytesIO(raw), "rb") as stream:
        _require(stream.getnchannels() == 1 and stream.getsampwidth() == 2 and stream.getcomptype() == "NONE", "Require canonical mono PCM16 WAV")
        _require(stream.getframerate() == binding[1] and stream.getnframes() == binding[2], "Actual audio clock differs")
        _require(len(stream.readframes(binding[2])) == binding[2] * 2, "Truncated WAV payload")
    cache[key] = binding, raw
    return raw


def _reviewers(reviewers):
    if reviewers is None:
        reviewers = {"reviewer_1": {"origin": "human", "judge_identity": None},
                     "reviewer_2": {"origin": "human", "judge_identity": None}}
    _require(isinstance(reviewers, dict) and bool(reviewers), "reviewers must map IDs to origin/judge_identity")
    sample = make_review_item("sample", "source", "candidate", "Action.")
    for reviewer_id, identity in reviewers.items():
        _require(isinstance(identity, dict) and set(identity) == {"origin", "judge_identity"}, "Invalid reviewer identity")
        review_template(sample, reviewer_id, **identity)
    return copy.deepcopy(reviewers)


def _html(items, assets, template_paths, phase):
    escape = lambda value: html.escape(str(value), quote=True)
    sections = []
    for item in items:
        source = assets[item["audio_sha256"]]["path"]
        reference = ("<h4>Training-reference subset (not unique choreography truth)</h4><pre>" + escape(item["reference_text"]) + "</pre>") if phase == "reference" else ""
        sections.append("<section><h2>" + escape(item["case_id"]) + " / " + escape(item["method_id"]) +
                        "</h2><p>Generation status: " + escape(item["generation_status"]) +
                        "</p><audio controls preload=\"none\" src=\"" + escape(source) +
                        "\"></audio><pre>" + escape(item["text"]) + "</pre>" + reference + "</section>")
    links = " ".join('<a href="' + escape(path) + '">' + escape(reviewer_id) + " JSON template</a>" for reviewer_id, path in template_paths.items())
    return ("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; media-src 'self'; style-src 'unsafe-inline'\">"
            "<title>Blind choreography review</title><style>body{max-width:960px;margin:2em auto;padding:0 1em;font-family:sans-serif}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere}section{border-top:1px solid #ccc;padding:1em 0}audio{width:100%}</style>"
            "<h1>Blind choreography review</h1><p>Candidate labels are randomized separately for each case and are NOT consistent method IDs. "
            "Listen to the bound audio and inspect the exact text. Copy your JSON template OUTSIDE this immutable bundle, then edit the copy. This page does not save scores. "
            "Keep unknown and not_applicable distinct. Hashes and listening ranges are declarations, not proof of listening.</p>"
            + ("<p>REFERENCE-ONLY PRIVATE STAGE: do not re-score quality after seeing the reference; additional plausible body actions are not automatically errors.</p>" if phase == "reference" else
               "<p>QUALITY STAGE: no training reference is included. Different choreographies can all be reasonable.</p>")
            + "<p>" + links + "</p>" + "".join(sections) + "</html>")


def _segment_identity(identity):
    value=copy.deepcopy(identity)
    entry=value.get('artifact_identity',{}).get('inference_adapter_source')
    if entry is not None:
        digest=entry.get('sha256')
        _require(isinstance(digest,str) and len(digest)==64, 'Source relocation lacks SHA binding')
        entry.pop('path',None)
    return value


def _same_segment_provider(left,right):
    left,right=copy.deepcopy(left),copy.deepcopy(right)
    for value in (left,right):
        source=value.get('artifact_identity',{}).get('inference_adapter_source')
        if source is not None:
            _require(set(source)=={'path','sha256','size_bytes'}, 'Unknown adapter identity fields')
            _require(isinstance(source['path'],str) and source['path'], 'Adapter source path missing')
            _require(isinstance(source['sha256'],str) and len(source['sha256'])==64, 'Adapter hash missing')
            _require(type(source['size_bytes']) is int and source['size_bytes']>0, 'Adapter byte size missing')
            source.pop('path')
    return object_sha(left)==object_sha(right)


def collect_review_bundle(result_paths, output_dir, *, references_path=None, reviewers=None,
                          seed=None, phase="quality", allow_segments=False):
    """Collect identical multi-method inventories; never omit or repair failures.

    Baseline seals/records/contracts and audio snapshots are checked. This is not
    an independent replay/audit of source model execution or its raw call chain.
    References supplied in quality mode remain private and are NOT in its items.
    Explicit reference mode publishes its materials only in private/reference/.
    """
    _require(isinstance(result_paths, (list, tuple)) and bool(result_paths), "Provide baseline result paths")
    _require(phase in ("quality", "reference"), "phase must be quality or reference")
    if seed is None:
        seed = secrets.randbits(63)
    _require(type(seed) is int and 0 <= seed < 2 ** 63, "seed must be integer in [0,2**63)")
    _require(phase != "reference" or references_path is not None, "Reference stage requires explicit private references")
    roster = _reviewers(reviewers)
    output = Path(output_dir).expanduser().resolve()
    _require(not output.exists(), "Review bundle output must be a new directory")
    sources, methods, contracts, synthetic, shared = {}, {}, [], False, None
    for raw_path in result_paths:
        path = Path(raw_path).expanduser().resolve(strict=True)
        method, rows, contract, result = _load_result(path, sources)
        if method in methods:
            _require(allow_segments, 'Duplicate method result')
            previous=methods[method]
            _require(not (set(previous['rows']) & set(rows)), 'Overlapping method segments')
            _require(object_sha(_segment_identity(previous['contract']['provider_identity']))==object_sha(_segment_identity(contract['provider_identity'])), 'Method provider content differs across segments')
            for field in ('seed','max_output_tokens','provider_config_snapshot','code_sha256'):
                _require(object_sha(previous['contract'][field])==object_sha(contract[field]),
                         'Method generation contract differs across segments: '+field)
            _require(all(Path(r['audio']['path']).is_absolute() for r in rows.values()), 'Segment audio paths must be absolute')
            previous['rows'].update(rows)
            for cid in rows:
                previous['case_provenance'][cid]=(object_sha(contract),object_sha(result))
        else:
            methods[method]={'rows':rows,'root':path.parent,'contract':contract,
                'case_provenance':{cid:(object_sha(contract),object_sha(result)) for cid in rows}}
        contracts.append(contract)
        synthetic = synthetic or result['synthetic']
    shared=None
    for method,source in methods.items():
        binding={cid:(row['source_group_id'],row['split'],_clock(row)) for cid,row in source['rows'].items()}
        _require(shared is None or binding==shared,'Methods differ in complete case/audio inventory')
        shared=binding
    first = next(iter(methods.values()))["rows"]
    refs = _reference_map(references_path, first, contracts, sources)
    rng = random.Random(seed)
    case_order, group_order = sorted(first), sorted({row["source_group_id"] for row in first.values()})
    rng.shuffle(case_order)
    rng.shuffle(group_order)
    groups = {group: "source_{:04d}".format(index + 1) for index, group in enumerate(group_order)}
    items, entries, assets, raw_assets, audio_cache = [], [], {}, {}, {}
    for case_index, cid in enumerate(case_order):
        order = sorted(methods)
        rng.shuffle(order)
        for method_index, method in enumerate(order):
            source = methods[method]
            row = source["rows"][cid]
            raw = _audio_bytes(row, source["root"], audio_cache)
            audio_sha = row["audio_sha256"]
            raw_assets[audio_sha] = raw
            assets[audio_sha] = {"path": "audio/" + audio_sha + ".wav", "sha256": audio_sha,
                                 "sample_rate": row["audio"]["sample_rate"], "num_samples": row["audio"]["num_samples"],
                                 "duration_seconds": row["duration_seconds"]}
            status = ("unknown" if row["status"] == "unknown" else
                      "truncated" if row.get("stop_reason") in ("length", "budget_truncation", "max_tokens")
                      else "ok" if row["status"] == "complete" else row["status"])
            reference = refs[cid]["action_reference"] if phase == "reference" else None
            blind = make_review_item("case_{:04d}".format(case_index + 1), groups[row["source_group_id"]],
                                     "candidate_{:02d}".format(method_index + 1), row["text"],
                                     audio_sha256=audio_sha, duration_seconds=row["duration_seconds"],
                                     generation_status=status, reference_text=reference,
                                     provenance={"synthetic": synthetic, "review_phase": phase})
            real = make_review_item(cid, row["source_group_id"], method, row["text"],
                                    audio_sha256=audio_sha, duration_seconds=row["duration_seconds"],
                                    generation_status=status, reference_text=reference,
                                    provenance={"synthetic": synthetic, "review_phase": phase,
                                                "baseline_contract_sha256": source["case_provenance"][cid][0],
                                                "baseline_results_sha256": source["case_provenance"][cid][1],
                                                "baseline_row_sha256": row["record_sha256"]})
            items.append(blind)
            entries.append({"blind_item_sha256": blind["item_sha256"], "true_item": real})
    for path, digest in sources.items():
        _require(_bytes_sha(Path(path).read_bytes()) == digest, "Input changed during review collection")
    for path, (binding, _) in audio_cache.items():
        _require(_bytes_sha(Path(path).read_bytes()) == binding[0], "Audio changed during review collection")
    stage_relative = "public" if phase == "quality" else "private/reference"
    stage = output / stage_relative
    output.mkdir(parents=True, exist_ok=False)
    stage.mkdir(parents=True, exist_ok=False)
    inventory = {}
    def publish(relative, value, raw=False):
        target = stage / relative
        if raw:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(value)
        else:
            write_json(target, value)
        inventory[relative] = _bytes_sha(target.read_bytes())
    for digest, raw in sorted(raw_assets.items()):
        publish("audio/" + digest + ".wav", raw, raw=True)
    publish("items.json", {"items": items})
    publish("rubric.json", rubric())
    publish("audio_assets.json", {"assets": list(assets.values())})
    template_paths = {}
    for reviewer_id, identity in sorted(roster.items()):
        path = "reviews/" + hashlib.sha256(reviewer_id.encode("utf-8")).hexdigest() + ".json"
        templates = [review_template(bound, reviewer_id, **identity) for bound in items]
        if phase == "reference":
            for template in templates:
                template["ratings"] = {dimension: {"status": "not_applicable", "score": None,
                                                   "reason": "Reference-only phase; do not re-score quality."} for dimension in DIMENSIONS}
        publish(path, {"reviews": templates})
        template_paths[reviewer_id] = path
    publish("index.html", _html(items, assets, template_paths, phase).encode("utf-8"), raw=True)
    mapping = _sealed({"schema": "captionlab.review_bundle_mapping.v1", "phase": phase, "seed": seed,
                       "stage": stage_relative, "inventory": inventory, "reviewers": roster,
                       "entries": entries, "sources": sources, "private_references": refs,
                       "synthetic": synthetic})
    write_json(output / "private" / "mapping.json", mapping)
    manifest = _sealed({"schema": "captionlab.review_bundle.v1", "phase": phase, "stage": stage_relative,
                        "case_count": len(first), "method_count": len(methods), "item_count": len(items),
                        "synthetic": synthetic, "quality_verified": False, "inventory": inventory,
                        "mapping_sha256": object_sha(mapping),
                        "audit_scope": "Sealed baseline summaries/rows/contracts/records and actual audio bytes; source raw calls/model execution not independently replayed.",
                        "sharing": "Share only public/ for quality. Never share private/ or the organizer bundle root."})
    write_json(output / "manifest.json", manifest)
    return manifest


def unblind_reviews(bundle_dir, review_paths, *, n_resamples=2000, seed=0,
                    method_a=None, method_b=None):
    """Validate returned blind reviews, rebind privately, then summarize by method.

    This function reads only; it never modifies templates or source run artifacts.
    Returned items/reviews contain private identities and must stay with organizer.
    """
    root = Path(bundle_dir).expanduser().resolve(strict=True)
    manifest = _verify(_snapshot(root / "manifest.json", {}))
    mapping = _verify(_snapshot(root / "private/mapping.json", {}))
    _require(manifest.get("schema") == "captionlab.review_bundle.v1" and mapping.get("schema") == "captionlab.review_bundle_mapping.v1", "Unknown review bundle schema")
    _require(object_sha(mapping) == manifest.get("mapping_sha256") and mapping.get("inventory") == manifest.get("inventory"), "Private mapping/public inventory binding mismatch")
    _require(mapping["phase"] == manifest["phase"] and mapping["stage"] == manifest["stage"] and mapping["synthetic"] is manifest["synthetic"], "Bundle stage/synthetic binding mismatch")
    expected_stage = "public" if manifest["phase"] == "quality" else "private/reference" if manifest["phase"] == "reference" else None
    _require(expected_stage is not None and manifest["stage"] == expected_stage, "Invalid review phase path")
    stage = root / expected_stage
    actual_files = {str(path.relative_to(stage)).replace("\\", "/") for path in stage.rglob("*") if path.is_file()}
    _require(actual_files == set(manifest["inventory"]), "Published review file inventory changed")
    for relative, digest in manifest["inventory"].items():
        path = (stage / relative).resolve(strict=True)
        _require(path.is_relative_to(stage.resolve()) and _bytes_sha(path.read_bytes()) == digest, "Published review asset changed or escapes its stage")
    doc = _snapshot(stage / "items.json", {})
    blind_items = doc.get("items")
    _require(isinstance(blind_items, list) and len(blind_items) == manifest["item_count"], "Blind item inventory mismatch")
    by_sha = {bound["item_sha256"]: bound for bound in blind_items}
    entries = mapping["entries"]
    _require(len(by_sha) == len(blind_items) and len(entries) == len(blind_items), "Duplicate/missing blind mapping entries")
    originals = {}
    for entry in entries:
        sha = entry["blind_item_sha256"]
        _require(sha in by_sha and sha not in originals, "Unknown or duplicate blind mapping")
        real = entry["true_item"]
        for field in ("text", "audio_sha256", "duration_seconds", "generation_status", "reference_text"):
            _require(object_sha(real[field]) == object_sha(by_sha[sha][field]), "Mapped true output differs from blind output")
        originals[sha] = real
    _require(isinstance(review_paths, (list, tuple)), "review_paths must be a list")
    normalized, bindings, seen, sources = [], [], set(), {}
    for path in review_paths:
        doc = _snapshot(path, sources)
        rows = doc.get("reviews") if isinstance(doc, dict) else doc
        _require(isinstance(rows, list), "Review file must contain reviews list")
        for row in rows:
            _require(isinstance(row, dict) and isinstance(row.get("item_sha256"), str) and row["item_sha256"] in by_sha, "Review names unknown blind item")
            blind = validate_review(row, by_sha[row["item_sha256"]])
            reviewer_id = blind["reviewer_id"]
            _require(reviewer_id in mapping["reviewers"] and mapping["reviewers"][reviewer_id] == {"origin": blind["origin"], "judge_identity": blind["judge_identity"]}, "Review roster identity mismatch")
            key = blind["item_sha256"], reviewer_id
            _require(key not in seen, "Duplicate returned item/reviewer")
            seen.add(key)
            if manifest["phase"] == "reference":
                _require(all(value["status"] == "not_applicable" for value in blind["ratings"].values()), "Reference stage cannot re-score quality")
            real = originals[blind["item_sha256"]]
            converted = copy.deepcopy(blind)
            converted.pop("review_sha256")
            converted["item_sha256"] = real["item_sha256"]
            converted = validate_review(converted, real)
            normalized.append(converted)
            bindings.append({"blind_item_sha256": blind["item_sha256"], "blind_review_sha256": blind["review_sha256"],
                             "true_item_sha256": real["item_sha256"], "true_review_sha256": converted["review_sha256"]})
    items = [originals[bound["item_sha256"]] for bound in blind_items]
    summary = summarize_quality(items, normalized, expected_reviewer_ids=mapping["reviewers"],
                                method_a=method_a, method_b=method_b, n_resamples=n_resamples, seed=seed)
    return {"schema": "captionlab.unblinded_quality.v1", "phase": manifest["phase"],
            "bundle_manifest_sha256": object_sha(manifest), "synthetic": manifest["synthetic"],
            "review_source_files": sources, "items": items, "reviews": normalized,
            "review_bindings": bindings, "summary": summary,
            "note": "Private organizer result. Origin labels, evidence_scope and baseline seals do not authenticate independent human/model execution."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("collect")
    command.add_argument("--result", action="append", required=True)
    command.add_argument("--references")
    command.add_argument("--reviewers", help="JSON reviewer origin/identity mapping")
    command.add_argument("--phase", choices=("quality", "reference"), default="quality")
    command.add_argument("--seed", type=int)
    command.add_argument("--out-dir", required=True)
    command = commands.add_parser("unblind")
    command.add_argument("--bundle", required=True)
    command.add_argument("--review", action="append", default=[])
    command.add_argument("--method-a")
    command.add_argument("--method-b")
    command.add_argument("--resamples", type=int, default=2000)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "collect":
            reviewers = _snapshot(args.reviewers, {}) if args.reviewers else None
            result = collect_review_bundle(args.result, args.out_dir, references_path=args.references,
                                           reviewers=reviewers, seed=args.seed, phase=args.phase)
            print(json.dumps({"phase": result["phase"], "cases": result["case_count"],
                              "methods": result["method_count"], "synthetic": result["synthetic"]}))
        else:
            _require(not Path(args.out).expanduser().resolve().is_relative_to((Path(args.bundle).expanduser().resolve() / "public")),
                     "Unblinded results contain private identities and must not be written into public/")
            result = unblind_reviews(args.bundle, args.review, n_resamples=args.resamples,
                                     seed=args.seed, method_a=args.method_a, method_b=args.method_b)
            write_json(args.out, result)
        return result
    except (ValueError, TypeError, KeyError, OSError, EOFError, wave.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
