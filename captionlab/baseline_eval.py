"""Raw, fixed-window audio baseline calls; no reference labels or caption repair.

This runner measures neither choreography quality nor source-group independence.
Completed results are reconstructed from immutable call evidence on resume.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from choreobench.audio import inspect_audio
from choreobench.baseline import _Journal, _journal_accounting, _recorded_result, _sealed, _verify
from choreobench.io import file_lock, file_sha, object_sha, read_json, read_json_snapshot, write_json
from choreobench.providers import create_provider


MANIFEST_SCHEMA = "captionlab.baseline_manifest.v1"


def _text(value, name):
    if (not isinstance(value, str) or not value.strip() or len(value) > 512
            or any(ord(c) < 32 for c in value)):
        raise ValueError("Require bounded nonempty " + name)
    return value


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _prepare(manifest_path):
    manifest, digest = read_json_snapshot(manifest_path)
    if (not isinstance(manifest, dict) or set(manifest) - {"schema", "dataset", "selection", "cases", "synthetic"}
            or not {"schema", "dataset", "selection", "cases"} <= set(manifest)
            or manifest["schema"] != MANIFEST_SCHEMA
            or type(manifest.get("synthetic", False)) is not bool
            or not isinstance(manifest["dataset"], dict) or not manifest["dataset"]
            or not isinstance(manifest["selection"], dict)
            or not isinstance(manifest["cases"], list) or not manifest["cases"]):
        raise ValueError("Invalid baseline manifest envelope")
    split = _text(manifest["selection"].get("split"), "selection split")
    seen, cases = set(), []
    for case in manifest["cases"]:
        if not isinstance(case, dict) or set(case) != {"case_id", "source_group_id", "split", "audio", "duration_seconds"}:
            raise ValueError("Baseline cases require explicit audio-only case fields")
        identifier = _text(case["case_id"], "case_id")
        if identifier in seen:
            raise ValueError("Duplicate case_id")
        seen.add(identifier)
        _text(case["source_group_id"], "source_group_id")
        if case["split"] != split:
            raise ValueError("Case split differs from frozen selection split")
        duration = case["duration_seconds"]
        audio = case["audio"]
        if (type(duration) not in (int, float) or not math.isfinite(duration) or duration != 5.0625
                or not isinstance(audio, dict) or set(audio) != {"path", "sha256", "sample_rate", "num_samples"}
                or not isinstance(audio["path"], str) or not audio["path"].strip() or not _sha(audio["sha256"])
                or type(audio["sample_rate"]) is not int or not 1 <= audio["sample_rate"] <= 192000
                or type(audio["num_samples"]) is not int
                or audio["num_samples"] != (81 * audio["sample_rate"] + 8) // 16):
            raise ValueError("Require materialized complete 5.0625s audio and an integer clock")
        path = Path(audio["path"])
        path = path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()
        actual = inspect_audio(path)
        if any(actual[k] != audio[k] for k in ("sha256", "sample_rate", "num_samples")):
            raise ValueError("Canonical audio bytes or clock differ from manifest")
        canonical = {k: actual[k] for k in ("path", "sha256", "sample_rate", "num_samples")}
        cases.append(dict(copy.deepcopy(case), audio=canonical))
    return manifest, digest, cases


def _code():
    root = Path(__file__).resolve().parents[1]
    names = ("captionlab/baseline_eval.py", "choreobench/audio.py", "choreobench/baseline.py",
             "choreobench/providers.py", "choreobench/io.py", "choreobench/resident.py")
    return {name: file_sha(root / name) for name in names}


def _request(case, method_id, seed, max_output_tokens):
    return dict(call_id=method_id + "/" + case["case_id"], response_kind="local_caption",
                audio=copy.deepcopy(case["audio"]), seed=seed, max_output_tokens=max_output_tokens)


def _accounting(journal):
    value = _journal_accounting(journal)
    value["costs"].sort(key=lambda item: item["call_id"])
    for key in ("unknown_outcome_call_ids", "unknown_usage_call_ids"):
        value[key].sort()
    return value


def _provider_source_unchanged(identity):
    # Replay answers are actual local inputs, not just descriptive metadata.
    if identity.get("kind") == "replay":
        path, digest = identity.get("path"), identity.get("sha256")
        if not isinstance(path, str) or not _sha(digest) or file_sha(path) != digest:
            raise ValueError("Replay source bytes changed")


def _row(case, request, journal, identity, method_id):
    call_id = request["call_id"]
    start, raw = journal.started[call_id], journal.records[call_id]
    if (object_sha(start["request"]) != object_sha(request)
            or object_sha(start.get("provider_identity")) != object_sha(identity)):
        raise ValueError("Historical request or provider identity changed")
    result = raw.get("result")
    # An incorrect returned identity is an integrity failure, not a quality failure.
    if raw.get("success") is True and (not isinstance(result, dict)
            or object_sha(result.get("identity")) != object_sha(identity)
            or result.get("synthetic") is not identity["synthetic"]):
        raise ValueError("Returned provider identity differs from frozen identity")
    error = None
    try:
        _recorded_result(raw, identity, request)
        status = "complete"
    except (ValueError, TypeError, KeyError) as failure:
        # Do not stringify a dict: JSON publication sorts its keys, so a loaded
        # exception dict can have a different repr from the original live one.
        error = (dict(type="RecordedProviderFailure", recorded_error=copy.deepcopy(raw.get("error")))
                 if raw.get("success") is not True else dict(type=type(failure).__name__, message=str(failure)))
        status = "unknown" if (raw.get("success") is not True or not isinstance(result, dict)
            or result.get("stop_reason") in (None, "http_error", "malformed_api_response", "infrastructure_error")) else "failed"
    return _sealed(dict(case_id=case["case_id"], source_group_id=case["source_group_id"],
        split=case["split"], method_id=method_id, call_id=call_id, status=status,
        text=result.get("text", "") if isinstance(result, dict) and isinstance(result.get("text"), str) else "",
        audio_sha256=case["audio"]["sha256"], audio=copy.deepcopy(case["audio"]),
        duration_seconds=case["duration_seconds"], stop_reason=result.get("stop_reason") if isinstance(result, dict) else None,
        started_record_sha256=start["record_sha256"], raw_record_sha256=raw["record_sha256"],
        error=error, quality_verified=False))


def _inventory(journal, requests, *, complete):
    expected = {r["call_id"] for r in requests}
    if set(journal.started) - expected or set(journal.records) - expected:
        raise ValueError("Unexpected journal call")
    if set(journal.started) != set(journal.records):
        raise ValueError("Pending/unknown call; automatic retry is forbidden")
    names = {journal._name(c) + suffix for c in journal.started for suffix in (".started.json", ".raw.json")}
    if {p.name for p in journal.directory.iterdir()} != names:
        raise ValueError("Journal file inventory changed")
    if complete and set(journal.records) != expected:
        raise ValueError("Completed result has missing calls")


def _lines(rows):
    return ("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                    for row in rows)).encode("utf-8")


def _publish_bytes(path, raw):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(str(temporary), str(path))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _summary(contract, rows, journal):
    return _sealed(dict(schema="captionlab.baseline_results.v1", complete=True,
        contract_sha256=object_sha(contract), method_id=contract["method_id"],
        case_count=len(rows), calls_used=len(journal.started),
        complete_count=sum(r["status"] == "complete" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        unknown_count=sum(r["status"] == "unknown" for r in rows), rows=rows,
        records_sha256=hashlib.sha256(_lines(rows)).hexdigest(),
        synthetic=contract["synthetic"], quality_verified=False,
        source_group_independence_verified=False, unique_audio_count=len({r["audio_sha256"] for r in rows}),
        scientific_result_status="synthetic_development_only" if contract["synthetic"] else "declared_real_not_independently_audited",
        **_accounting(journal)))


def run_baseline(manifest_path, provider_config_path, method_id, out_dir, *, seed=42,
                 max_output_tokens=1024, resume=False, provider_factory=None):
    """One call per frozen case; raw failed/truncated answers are never replaced.

    ``provider_factory(spec, role=..., base_dir=...)`` is an explicit CPU-test seam.
    A complete resume validates all artifacts before creating any provider.
    """
    manifest_path, provider_config_path, output = [Path(p).resolve() for p in (manifest_path, provider_config_path, out_dir)]
    _text(method_id, "method_id")
    if (type(seed) is not int or not 0 <= seed < 2**63 or type(max_output_tokens) is not int
            or not 0 < max_output_tokens <= 16384 or type(resume) is not bool):
        raise ValueError("Invalid seed, output budget or resume flag")
    manifest, manifest_sha, cases = _prepare(manifest_path)
    spec, spec_sha = read_json_snapshot(provider_config_path)
    if not isinstance(spec, dict) or spec.get("kind") not in ("native", "native_service", "replay"):
        raise ValueError("Explicit native, native_service or replay provider required; no mock/text fallback")
    base = dict(schema="captionlab.baseline_contract.v1", method_id=method_id, seed=seed,
        max_output_tokens=max_output_tokens, manifest_path=str(manifest_path), manifest_sha256=manifest_sha,
        manifest_snapshot=manifest, provider_config_path=str(provider_config_path), provider_config_sha256=spec_sha,
        provider_config_snapshot=spec, cases=cases, code_sha256=_code(),
        input_contract="audio_only_fixed_5.0625s_no_reference_or_style_labels", generation_calls_per_case=1)
    requests = [_request(case, method_id, seed, max_output_tokens) for case in cases]
    if output.exists() and not resume:
        raise FileExistsError(str(output))
    if resume and (not output.is_dir() or not (output / "contract.json").is_file() or not (output / "calls").is_dir()):
        raise ValueError("Resume requires existing contract and call directory")
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    with file_lock(output / ".run.lock"):
        provider = None
        if resume:
            contract = _verify(read_json(output / "contract.json"))
            expected_keys = set(base) | {"provider_identity", "synthetic", "record_sha256"}
            if set(contract) != expected_keys or any(object_sha(contract.get(k)) != object_sha(v) for k, v in base.items()):
                raise ValueError("Resume contract/input/code identity changed")
            identity = contract["provider_identity"]
        else:
            factory = provider_factory or create_provider
            provider = factory(copy.deepcopy(spec), role="local", base_dir=provider_config_path.parent)
            identity = copy.deepcopy(provider.identity)
            if not isinstance(identity, dict) or type(identity.get("synthetic")) is not bool:
                raise ValueError("Provider requires an explicit synthetic identity")
            contract = _sealed(dict(base, provider_identity=identity,
                synthetic=manifest.get("synthetic", False) or identity["synthetic"]))
            write_json(output / "contract.json", contract)
        if (not isinstance(identity, dict) or type(identity.get("synthetic")) is not bool
                or contract["synthetic"] is not (manifest.get("synthetic", False) or identity["synthetic"])):
            raise ValueError("Malformed frozen provider/synthetic declaration")
        _provider_source_unchanged(identity)
        journal = _Journal(output, object_sha(contract), len(cases))
        complete = (output / "results.json").exists()
        _inventory(journal, requests, complete=complete)
        rows = []
        for case, request in zip(cases, requests):
            if request["call_id"] in journal.records:
                row = _row(case, request, journal, identity, method_id)
                artifact = output / "cases" / (journal._name(request["call_id"]) + ".json")
                if artifact.exists():
                    if object_sha(read_json(artifact)) != object_sha(row):
                        raise ValueError("Published case artifact differs from raw evidence")
                elif complete:
                    raise ValueError("Completed result is missing its case artifact")
                rows.append(row)
        if complete:
            summary = _summary(contract, rows, journal)
            if (object_sha(read_json(output / "results.json")) != object_sha(summary)
                    or not (output / "records.jsonl").is_file()
                    or (output / "records.jsonl").read_bytes() != _lines(rows)):
                raise ValueError("Published results/records differ from raw evidence")
            expected_cases = {journal._name(request["call_id"]) + ".json" for request in requests}
            if {path.name for path in (output / "cases").iterdir()} != expected_cases:
                raise ValueError("Published case artifact inventory changed")
            return summary
        if (output / "records.jsonl").exists():
            raise ValueError("Incomplete publication exists; inspect it without resampling")
        if provider is None:
            factory = provider_factory or create_provider
            provider = factory(copy.deepcopy(spec), role="local", base_dir=provider_config_path.parent)
        def unchanged(current_case=None):
            if object_sha(provider.identity) != object_sha(identity):
                raise ValueError("Provider identity drift; further dispatch forbidden")
            _provider_source_unchanged(identity)
            for path, digest in ((manifest_path, manifest_sha), (provider_config_path, spec_sha)):
                if file_sha(path) != digest:
                    raise ValueError("Input bytes changed during run")
            for case in cases if current_case is None else (current_case,):
                if file_sha(case["audio"]["path"]) != case["audio"]["sha256"]:
                    raise ValueError("Audio bytes changed during run")
        rows = []
        for case, request in zip(cases, requests):
            unchanged(case)
            if request["call_id"] not in journal.records:
                try:
                    journal.call(provider, request)
                except (ValueError, TypeError, KeyError):
                    if request["call_id"] not in journal.records:
                        raise  # Never turn an unsealed/unknown outcome into a new call.
            unchanged(case)
            row = _row(case, request, journal, identity, method_id)
            artifact = output / "cases" / (journal._name(request["call_id"]) + ".json")
            if artifact.exists():
                if object_sha(read_json(artifact)) != object_sha(row):
                    raise ValueError("Published case artifact changed")
            else:
                write_json(artifact, row)
            rows.append(row)
        _inventory(journal, requests, complete=True)
        unchanged()
        summary = _summary(contract, rows, journal)
        _publish_bytes(output / "records.jsonl", _lines(rows))
        write_json(output / "results.json", summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--provider-config", required=True, type=Path)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = run_baseline(args.manifest, args.provider_config, args.method_id, args.out_dir,
                          seed=args.seed, max_output_tokens=args.max_output_tokens, resume=args.resume)
    print(json.dumps({k: result[k] for k in ("case_count", "complete_count", "failed_count", "unknown_count", "synthetic")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
