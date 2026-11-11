"""Small explicit sol61 fixtures, without HF traffic, weights or training."""
import copy
from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import wave

import pytest

from captionlab import sol61
from captionlab.io import file_sha, object_sha, read_json, text_sha, write_json


def record(ident, split, group, gate=True):
    return dict(id=ident, split=split, source_group_id=group, gate=gate,
        motion_caption="Dance style: test-style\nPRIVATE-MOVEMENT-" + ident + ". Repeat the same movement.",
        dance_style="test-style", caption_version="sol61", gestures=[dict(event_id="w0e0")],
        video_path="private-video/" + ident + ".mp4")


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def sources(tmp_path):
    root = tmp_path / sol61.SUBDIR; root.mkdir()
    rows = dict(train=[record("train0", "train", "train-group")],
        val=[record("val0", "val", "group0"), record("val1", "val", "group0"),
             record("val2", "val", "group1"), record("val3", "val", "group2", False)],
        test=[record("test0", "test", "test-group")])
    for split, values in rows.items(): jsonl(root / (split + "_captions.jsonl"), values)
    return root, rows


def receipt(root, *, revision=sol61.REVISION, repo=sol61.REPO, legacy=False):
    value = dict(repo_id=repo, revision=revision, files={
        sol61.SUBDIR + "/" + split + "_captions.jsonl": file_sha(root / (split + "_captions.jsonl"))
        for split in sol61.SPLITS})
    if not legacy: value["schema"] = sol61.DOWNLOAD_SCHEMA
    write_json(root.parent / "download_receipt.json", value)
    return value


def test_audit_counts_and_unreceipted_revision_is_only_declared(sources):
    root, _ = sources
    report, metadata, bad_ids, bad_groups = sol61.audit_sources(root)
    assert {k: v["records"] for k, v in report["splits"].items()} == dict(train=1, val=4, test=1)
    assert report["splits"]["val"]["source_groups"] == 3
    assert report["cross_split_id_count"] == report["cross_split_group_count"] == 0
    assert not bad_ids and not bad_groups
    assert report["provenance"]["revision_status"] == "declared_revision_only"
    assert not report["provenance"]["independent_remote_revision_verification"]
    assert not report["song_or_near_duplicate_isolation_verified"]
    assert not report["actual_training_membership_verified"]
    assert set(metadata["val"][0]) == {"id", "source_group_id", "gate"}


@pytest.mark.parametrize("legacy", [False, True])
def test_receipt_binds_repo_revision_and_all_audited_files(sources, legacy):
    root, _ = sources; receipt(root, legacy=legacy)
    manifest, _, report = sol61.freeze_cases(root, split="val", limit=3)
    provenance = report["provenance"]
    assert provenance["revision_status"] == "download_receipt_verified"
    assert provenance["legacy_receipt"] is legacy
    assert provenance["receipt_sha256"] == file_sha(root.parent / "download_receipt.json")
    assert manifest["dataset"]["provenance"] == provenance
    assert not provenance["independent_remote_revision_verification"]


@pytest.mark.parametrize("problem", ["repo", "revision", "changed", "missing_entry", "unknown_field", "path", "missing_file"])
def test_invalid_receipt_never_silently_becomes_declared_only(sources, problem):
    root, _ = sources
    value = receipt(root)
    path = root.parent / "download_receipt.json"
    if problem == "repo": value["repo_id"] = "different/dataset"
    elif problem == "revision": value["revision"] = "a" * 40
    elif problem == "changed": (root / "val_captions.jsonl").write_bytes((root / "val_captions.jsonl").read_bytes() + b" ")
    elif problem == "missing_entry": value["files"].pop(sol61.SUBDIR + "/val_captions.jsonl")
    elif problem == "unknown_field": value["authenticated"] = True
    elif problem == "path": value["files"]["../escape"] = "0" * 64
    else: (root / "train_captions.jsonl").unlink()
    path.unlink(); write_json(path, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        sol61.freeze_cases(root, split="val", limit=1)


@pytest.mark.parametrize("problem", ["split", "duplicate_id", "header", "gate"])
def test_malformed_or_duplicate_rows_rejected(sources, problem):
    root, rows = sources
    if problem == "split": rows["val"][0]["split"] = "test"
    elif problem == "duplicate_id": rows["val"].append(copy.deepcopy(rows["val"][0]))
    elif problem == "header": rows["val"][0]["motion_caption"] = "Dance style: wrong\nMovement."
    else: rows["val"][0]["gate"] = 1
    jsonl(root / "val_captions.jsonl", rows["val"])
    with pytest.raises(ValueError): sol61.audit_sources(root)


def test_cross_split_ids_and_groups_are_reported_and_excluded_without_fallback(sources):
    root, rows = sources
    rows["train"] += [record("val2", "train", "different-group"), record("train1", "train", "group2")]
    jsonl(root / "train_captions.jsonl", rows["train"])
    report, _, bad_ids, bad_groups = sol61.audit_sources(root)
    assert bad_ids == {"val2"} and bad_groups == {"group2"}
    assert report["cross_split_id_count"] == report["cross_split_group_count"] == 1
    manifest, _, _ = sol61.freeze_cases(root, split="val", limit=1)
    assert manifest["cases"][0]["source_group_id"] == "group0"
    with pytest.raises(ValueError, match="no split fallback"):
        sol61.freeze_cases(root, split="val", limit=2)


def test_one_per_group_is_deterministic_not_file_order_and_gt_is_private(sources):
    root, rows = sources
    first, refs, _ = sol61.freeze_cases(root, split="val", limit=3, seed=42)
    chosen = [r["case_id"] for r in first["cases"]]
    assert len({r["source_group_id"] for r in first["cases"]}) == 3
    rows["val"].reverse(); jsonl(root / "val_captions.jsonl", rows["val"])
    second, _, _ = sol61.freeze_cases(root, split="val", limit=3, seed=42)
    assert [r["case_id"] for r in second["cases"]] == chosen
    assert "PRIVATE-MOVEMENT" not in json.dumps(first)
    assert "private-video" not in json.dumps(first)
    for case, reference in zip(first["cases"], refs["records"]):
        assert set(case) == {"case_id", "source_group_id", "split", "audio", "duration_seconds"}
        assert all(value is None for value in case["audio"].values())
        assert reference["case_id"] == case["case_id"]
        assert reference["raw_caption_sha256"] == text_sha(reference["raw_caption"])
        assert reference["action_reference"] == reference["raw_caption"].partition("\n")[2]
        assert reference["action_reference"].endswith("Repeat the same movement.")
        assert not reference["reference_is_unique_choreography_truth"]
    assert refs["selection_sha256"] == object_sha(first["selection"])


def test_gate_selection_explicit_and_train_never_used(sources):
    root, _ = sources
    manifest, _, _ = sol61.freeze_cases(root, split="val", limit=2, gate_policy="pass")
    assert {r["source_group_id"] for r in manifest["cases"]} == {"group0", "group1"}
    with pytest.raises(ValueError): sol61.freeze_cases(root, split="train", limit=1)
    with pytest.raises(ValueError): sol61.freeze_cases(root, split="val", limit=3, gate_policy="pass")
    for seed in (-1, True, 2**63):
        with pytest.raises(ValueError): sol61.freeze_cases(root, split="val", limit=1, seed=seed)


def write_wav(path, *, count=81000, rate=16000, channels=1, width=2, sample=0):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels); handle.setsampwidth(width); handle.setframerate(rate)
        handle.writeframes(bytes([sample]) * count * channels * width)


@pytest.fixture
def media(sources):
    root, _ = sources
    manifest, _, _ = sol61.freeze_cases(root, split="val", limit=3)
    audio_root = root.parent / "audio"; audio_root.mkdir()
    rows = []
    for n, case in enumerate(manifest["cases"]):
        wav = audio_root / (case["case_id"] + ".wav"); write_wav(wav, sample=n + 1)
        rows.append(dict(id=case["case_id"], source_group_id=case["source_group_id"], split="val",
            audio_sha256=file_sha(wav), audio_window_seconds=[0.0, 5.0625], audio_duration_seconds=5.0625,
            motion_caption="PRIVATE-ORIGINAL-GT-NOT-FOR-MODEL"))
    path = root.parent / "media.jsonl"; jsonl(path, rows)
    return manifest, path, audio_root, rows


def test_binding_joins_by_id_and_hash_and_keeps_private_media_out(media):
    manifest, path, audio_root, rows = media
    original = copy.deepcopy(manifest)
    jsonl(path, list(reversed(rows)))
    result = sol61.bind_audio(manifest, path, audio_root)
    assert manifest == original
    assert result["dataset"]["media_bound"]
    assert result["dataset"]["media_manifest_sha256"] == file_sha(path)
    assert result["dataset"]["selected_pcm_duplicates"] == 0
    for case in result["cases"]:
        audio = case["audio"]
        assert set(audio) == {"path", "sha256", "sample_rate", "num_samples"}
        assert audio["sha256"] == file_sha(audio_root / (case["case_id"] + ".wav"))
        assert audio["sample_rate"] == 16000 and audio["num_samples"] == 81000
    assert "PRIVATE-ORIGINAL-GT" not in json.dumps(result)
    assert sol61.bind_audio(result, path, audio_root) == result


@pytest.mark.parametrize("problem", ["split", "group", "sha", "window", "duration", "duplicate", "missing", "previous_clock", "gt"])
def test_binding_rejects_bad_identity_clock_or_private_case_fields(media, problem):
    manifest, path, audio_root, rows = media
    if problem == "split": rows[0]["split"] = "test"
    elif problem == "group": rows[0]["source_group_id"] = "other-source"
    elif problem == "sha": rows[0]["audio_sha256"] = "0" * 64
    elif problem == "window": rows[0]["audio_window_seconds"] = [1.0, 6.0625]
    elif problem == "duration": rows[0]["audio_duration_seconds"] = 5.0
    elif problem == "duplicate": rows.append(copy.deepcopy(rows[0]))
    elif problem == "missing": rows.pop()
    elif problem == "previous_clock": manifest["cases"][0]["audio"]["num_samples"] = 80000
    else: manifest["cases"][0]["reference"] = "PRIVATE TARGET"
    jsonl(path, rows)
    with pytest.raises(ValueError): sol61.bind_audio(manifest, path, audio_root)


@pytest.mark.parametrize("count,channels,width", [(80000, 1, 2), (81000, 2, 2), (81000, 1, 1)])
def test_binding_requires_actual_mono_pcm16_complete_waveform(media, count, channels, width):
    manifest, path, audio_root, rows = media
    wav = audio_root / (rows[0]["id"] + ".wav")
    write_wav(wav, count=count, channels=channels, width=width)
    rows[0]["audio_sha256"] = file_sha(wav); jsonl(path, rows)
    with pytest.raises(ValueError): sol61.bind_audio(manifest, path, audio_root)


def test_duplicate_pcm_detected_despite_different_wav_hashes(media):
    manifest, path, audio_root, rows = media
    a, b = [audio_root / (row["id"] + ".wav") for row in rows[:2]]
    b.write_bytes(a.read_bytes() + b"extra-container-bytes")
    assert file_sha(a) != file_sha(b)
    rows[1]["audio_sha256"] = file_sha(b); jsonl(path, rows)
    with pytest.raises(ValueError, match="identical PCM"):
        sol61.bind_audio(manifest, path, audio_root)


class RequestError(Exception):
    pass


def network(monkeypatch, *, fail_name=None):
    calls, state = [], {"failed": False}
    payloads = {name: ("pinned-content-for-" + name).encode("utf-8") for name in sol61._SOURCE_FILES + sol61._MEDIA_FILES}
    class Response:
        status_code = 200
        def __init__(self, name): self.name = name
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def iter_content(self, _):
            raw = payloads[self.name]; yield raw[:4]
            if self.name == fail_name and not state["failed"]:
                state["failed"] = True
                raise RequestError("token-or-CDN-URL-must-not-be-serialized")
            yield raw[4:]
    def get(url, **kwargs):
        name = url.split("/resolve/" + sol61.REVISION + "/", 1)[1]
        calls.append(name)
        assert kwargs["stream"] is True and kwargs["timeout"] == (20, 120)
        return Response(name)
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=get, RequestException=RequestError))
    return calls, payloads


def test_fetch_stream_interruption_resumes_only_unfinished_metadata(tmp_path, monkeypatch):
    calls, payloads = network(monkeypatch, fail_name=sol61._SOURCE_FILES[1])
    monkeypatch.setenv("HF_TOKEN", "SECRET-TOKEN")
    out = tmp_path / "download"
    with pytest.raises(RuntimeError) as caught:
        sol61.fetch_sources(out)
    assert "SECRET" not in str(caught.value) and "CDN" not in str(caught.value)
    saved = read_json(out / "download_receipt.json")
    assert set(saved["files"]) == {sol61._SOURCE_FILES[0]}
    result = sol61.fetch_sources(out)
    assert Counter(calls)[sol61._SOURCE_FILES[0]] == 1
    assert Counter(calls)[sol61._SOURCE_FILES[1]] == 2
    assert set(result["files"]) == set(sol61._SOURCE_FILES)
    for name in sol61._SOURCE_FILES:
        assert (out / name).read_bytes() == payloads[name]
        assert result["files"][name] == file_sha(out / name)
        assert not (out / (name + ".partial")).exists()
    before = len(calls)
    assert sol61.fetch_sources(out) == result and len(calls) == before
    extended = sol61.fetch_sources(out, include_media=True)
    assert set(extended["files"]) == set(sol61._SOURCE_FILES + sol61._MEDIA_FILES)
    assert len(calls) == before + 2
    assert "SECRET-TOKEN" not in (out / "download_receipt.json").read_text(encoding="utf-8")


def test_fetch_recovers_crash_after_receipt_before_publication(tmp_path, monkeypatch):
    calls, _ = network(monkeypatch)
    original_link = sol61.os.link
    def crash(*args, **kwargs): raise OSError("simulated publication crash")
    monkeypatch.setattr(sol61.os, "link", crash)
    out = tmp_path / "download"
    with pytest.raises(OSError): sol61.fetch_sources(out)
    name = sol61._SOURCE_FILES[0]
    assert read_json(out / "download_receipt.json")["files"][name] == file_sha(out / (name + ".partial"))
    assert not (out / name).exists()
    monkeypatch.setattr(sol61.os, "link", original_link)
    result = sol61.fetch_sources(out)
    assert Counter(calls)[name] == 1 and (out / name).is_file()
    assert set(result["files"]) == set(sol61._SOURCE_FILES)


@pytest.mark.parametrize("changed", [False, True])
def test_unreceipted_old_file_requires_matching_fresh_pinned_download(tmp_path, monkeypatch, changed):
    calls, payloads = network(monkeypatch)
    out = tmp_path / "download"; target = out / sol61._SOURCE_FILES[0]
    target.parent.mkdir(parents=True)
    raw = b"different-local-file" if changed else payloads[sol61._SOURCE_FILES[0]]
    target.write_bytes(raw)
    if changed:
        with pytest.raises(FileExistsError): sol61.fetch_sources(out)
        assert target.read_bytes() == raw and not (out / "download_receipt.json").exists()
    else:
        before = target.stat().st_mtime_ns
        result = sol61.fetch_sources(out)
        assert target.stat().st_mtime_ns == before
        assert result["files"][sol61._SOURCE_FILES[0]] == file_sha(target)
    assert calls[0] == sol61._SOURCE_FILES[0]


@pytest.mark.parametrize("problem", ["repo", "revision", "changed", "missing", "path"])
def test_fetch_existing_receipt_mismatch_rejected_without_network(tmp_path, monkeypatch, problem):
    calls, _ = network(monkeypatch); out = tmp_path / "download"
    sol61.fetch_sources(out); before = len(calls)
    path = out / "download_receipt.json"; value = read_json(path)
    if problem == "repo": value["repo_id"] = "other/dataset"
    elif problem == "revision": value["revision"] = "b" * 40
    elif problem == "changed": (out / sol61._SOURCE_FILES[0]).write_bytes(b"changed")
    elif problem == "missing": (out / sol61._SOURCE_FILES[0]).unlink()
    else: value["files"]["../outside"] = "0" * 64
    path.unlink(); write_json(path, value)
    with pytest.raises(ValueError): sol61.fetch_sources(out)
    assert len(calls) == before
