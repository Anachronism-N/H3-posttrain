"""Synthetic waveform checks for public structure evidence, not detector quality."""
import json
import copy
import hashlib
import math
import struct
import subprocess
import sys
import wave

import pytest

from choreobench.io import file_sha, object_sha, read_json, write_json
from choreobench.music_structure import DETECTIONS_SCHEMA, analyze_structure, wrap_legacy_detections
from choreobench.public_structure import load_public_structure


@pytest.fixture
def source(tmp_path):
    rate, count = 16000, 16000 * 65
    wav = tmp_path / "source.wav"
    def tone(frequency):
        return b"".join(struct.pack("<h", int(3000 * math.sin(2 * math.pi * frequency * i / rate))) for i in range(rate))
    pcm = tone(220) * 32 + tone(2200) * 33
    with wave.open(str(wav), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(rate); f.writeframes(pcm)
    song = dict(song_id="test", split="dev", song_group_id=None, artist_group_id=None, near_duplicate_group_id=None,
        brief="Synthetic spectral contrast fixture", audio=dict(path="source.wav", sha256=file_sha(wav), sample_rate=rate, num_samples=count),
        music=dict(source="unavailable", beats=[], downbeats=[], sections=[dict(section_id="a", start=0, end=count,
            label="unknown", repeat_group=None)]))
    dataset = tmp_path / "dataset.json"
    write_json(dataset, dict(schema="choreobench.dataset.v1", dataset_id="synthetic", synthetic=True,
        grouping=dict(status="unverified", evidence=""), songs=[song]))
    detections = tmp_path / "beats.jsonl"
    row = dict(schema=DETECTIONS_SCHEMA, id="test", status="ok", audio_sha256=file_sha(wav),
               sample_rate=rate, num_samples=count, time_unit="seconds", synthetic=True,
               detector=dict(name="synthetic_fixture", version="fixture-v1", model_sha256=None,
                             code_sha256="a" * 64, parameters={"fixture": True}),
               beats=list(range(1, 65)), downbeats=list(range(1, 65, 2)))
    detections.write_text(json.dumps(row) + "\n")
    return dataset, song, detections, row


def test_novelty_sidecar_preserves_clock_unknown_labels_and_source_dataset(source, tmp_path):
    pytest.importorskip("numpy"); pytest.importorskip("soundfile"); pytest.importorskip("scipy")
    dataset, song, detections, row = source
    before = file_sha(dataset)
    result = analyze_structure(dataset, detections, tmp_path / "structure.json")
    assert file_sha(dataset) == before
    music = result["songs"][0]["music"]
    assert music["sections"][0]["start"] == 0 and music["sections"][-1]["end"] == song["audio"]["num_samples"]
    assert all(s["label"] == "unknown" and s["repeat_group"] is None for s in music["sections"])
    assert all(a["end"] == b["start"] for a, b in zip(music["sections"], music["sections"][1:]))
    assert all(s["start"] in music["downbeats"] for s in music["sections"][1:])
    effective, snapshot, _ = load_public_structure(tmp_path / "structure.json", before, [song])
    assert effective[0]["music"] == music and snapshot == result
    evidence = read_json(tmp_path / "structure.analysis.json")["evidence"][0]
    assert not evidence["functional_labels_predicted"] and not evidence["threshold_calibrated"]
    assert not evidence["semantic_phrase_detected"] and not evidence["beat_accuracy_verified"]
    assert music["source"].startswith("synthetic_fixture@fixture-v1+spectral-novelty-v1:")
    assert evidence["detector"] == row["detector"]
    assert evidence["detector_identity_status"] == "declared_not_independently_authenticated"
    assert evidence["detection_record_sha256"] == hashlib.sha256(detections.read_bytes().splitlines()[0]).hexdigest()
    assert evidence["detection_synthetic"] and evidence["output_synthetic"]
    with pytest.raises(FileExistsError):
        analyze_structure(dataset, detections, tmp_path / "structure.json")


@pytest.mark.parametrize("field,value", [("audio_sha256", "b" * 64), ("status", "failed"),
    ("beats", [1, 1]), ("downbeats", [1, 70]), ("beats", [True])])
def test_bad_detection_input_is_explicitly_rejected(source, tmp_path, field, value):
    pytest.importorskip("numpy"); pytest.importorskip("soundfile"); pytest.importorskip("scipy")
    dataset, _, detections, row = source
    row[field] = value
    detections.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        analyze_structure(dataset, detections, tmp_path / "structure.json")
    assert not (tmp_path / "structure.json").exists()


def write_row(path, row):
    path.write_text(json.dumps(row, allow_nan=False) + "\n", encoding="utf-8")


@pytest.mark.parametrize("dataset_synthetic,detection_synthetic", [(False, True), (True, False), (True, True), (False, False)])
def test_taint_is_or_of_explicit_dataset_and_detector_declarations(source, tmp_path, dataset_synthetic, detection_synthetic):
    dataset, _, detections, row = source
    value = read_json(dataset)
    value["synthetic"] = dataset_synthetic  # Deliberate metadata fixture; no real audio/model result.
    write_json(dataset, value, overwrite=True)
    row["synthetic"] = detection_synthetic
    row["detector"]["name"] = "declared_detector"
    if not detection_synthetic:
        row["detector"]["model_sha256"] = "b" * 64
    write_row(detections, row)
    result = analyze_structure(dataset, detections, tmp_path / "structure.json")
    assert result["songs"][0]["synthetic"] is (dataset_synthetic or detection_synthetic)
    assert result["songs"][0]["music"]["source"].startswith("declared_detector@")
    assert "beatthis" not in result["songs"][0]["music"]["source"]


@pytest.mark.parametrize("mutation", ["missing_schema", "old_schema", "missing_synthetic", "bool_clock", "float_count",
    "wrong_rate", "wrong_count", "wrong_unit", "extra", "detector_extra", "bad_model", "real_model_missing",
    "bad_code", "missing_version", "bad_parameters", "duplicate_samples", "raw_downbeat_not_beat", "rounded_end"])
def test_strict_detection_schema_rejects_undeclared_identity_clock_or_added_fields(source, tmp_path, mutation):
    dataset, _, detections, row = source
    if mutation == "missing_schema": row.pop("schema")
    elif mutation == "old_schema": row["schema"] = "beatthis_audio_detections_v1"
    elif mutation == "missing_synthetic": row.pop("synthetic")
    elif mutation == "bool_clock": row["sample_rate"] = True
    elif mutation == "float_count": row["num_samples"] = float(row["num_samples"])
    elif mutation == "wrong_rate": row["sample_rate"] += 1
    elif mutation == "wrong_count": row["num_samples"] += 1
    elif mutation == "wrong_unit": row["time_unit"] = "milliseconds"
    elif mutation == "extra": row["semantic_goals"] = ["hidden"]
    elif mutation == "detector_extra": row["detector"]["unversioned_metadata"] = True
    elif mutation == "bad_model": row["detector"]["model_sha256"] = "NOT_A_SHA"
    elif mutation == "real_model_missing": row["synthetic"] = False
    elif mutation == "bad_code": row["detector"]["code_sha256"] = "A" * 64
    elif mutation == "missing_version": row["detector"].pop("version")
    elif mutation == "bad_parameters": row["detector"]["parameters"] = []
    elif mutation == "duplicate_samples": row.update(beats=[.001, .00101], downbeats=[])
    elif mutation == "raw_downbeat_not_beat": row.update(beats=[.001], downbeats=[.00101])
    else: row.update(beats=[64.999999], downbeats=[])
    write_row(detections, row)
    with pytest.raises(ValueError):
        analyze_structure(dataset, detections, tmp_path / "structure.json")
    assert not (tmp_path / "structure.json").exists()


@pytest.mark.parametrize("contents", ["duplicate_key", "nonfinite", "overflow", "blank", "duplicate_row", "unknown_row", "missing_row"])
def test_jsonl_snapshot_is_strict_and_inventory_is_exact(source, tmp_path, contents):
    dataset, _, detections, row = source
    raw = json.dumps(row)
    if contents == "duplicate_key": raw = raw[:-1] + ', "synthetic": false}'
    elif contents == "nonfinite": raw = raw.replace('"fixture": true', '"fixture": NaN')
    elif contents == "overflow": raw = raw.replace('"fixture": true', '"fixture": 1e999')
    elif contents == "blank": raw += "\n\n"
    elif contents == "duplicate_row": raw += "\n" + raw
    elif contents == "unknown_row": raw = raw.replace('"id": "test"', '"id": "other"')
    else: raw = ""
    detections.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError):
        analyze_structure(dataset, detections, tmp_path / "structure.json")
    assert not (tmp_path / "structure.json").exists()


def legacy_row(source):
    row = source[3]
    return {key: copy.deepcopy(row[key]) for key in ("id", "status", "audio_sha256", "beats", "downbeats")}


def test_explicit_legacy_wrapper_preserves_source_bytes_hashes_and_receipt_without_inference(source, tmp_path):
    dataset, _, detections, row = source
    old = legacy_row(source)
    old["original_private_metadata"] = "retained by the old bytes, not copied into the public row"
    write_row(detections, old)
    before = detections.read_bytes()
    receipt = tmp_path / "run_receipt.json"
    write_json(receipt, dict(output_sha256=file_sha(detections)))
    wrapped = tmp_path / "wrapped.jsonl"
    result = wrap_legacy_detections(dataset, detections, wrapped, detector=row["detector"], synthetic=True,
        time_unit="seconds", run_receipt_path=receipt)
    assert detections.read_bytes() == before
    assert result[0]["beats"] == old["beats"] and result[0]["downbeats"] == old["downbeats"]
    assert result[0]["source_binding"] == {"artifact_sha256": hashlib.sha256(before).hexdigest(),
        "record_sha256": hashlib.sha256(before.splitlines()[0]).hexdigest(), "run_receipt_sha256": file_sha(receipt)}
    assert result[0]["synthetic"] is True
    assert "original_private_metadata" not in wrapped.read_text(encoding="utf-8")
    analyze_structure(dataset, wrapped, tmp_path / "structure.json")
    evidence = read_json(tmp_path / "structure.analysis.json")["evidence"][0]
    assert evidence["detection_source_binding"] == result[0]["source_binding"]
    with pytest.raises(FileExistsError):
        wrap_legacy_detections(dataset, detections, wrapped, detector=row["detector"], synthetic=True, time_unit="seconds")


@pytest.mark.parametrize("mutation", ["synthetic", "model", "code", "clock", "audio_frames", "duration", "params", "detector", "failed"])
def test_wrapper_never_overrides_conflicting_old_declarations_or_filters_failures(source, tmp_path, mutation):
    dataset, _, detections, row = source
    old = legacy_row(source)
    if mutation == "synthetic": old["synthetic"] = False
    elif mutation == "model": old["model_sha256"] = "b" * 64
    elif mutation == "code": old["library_source_sha256"] = "b" * 64
    elif mutation == "clock": old["sample_rate"] = True
    elif mutation == "audio_frames": old["audio_frames"] = 100
    elif mutation == "duration": old["duration_seconds"] = 64
    elif mutation == "params": old["params"] = {"dbn": True}
    elif mutation == "detector": old["detector"] = "another_detector"
    else: old["status"] = "failed"
    write_row(detections, old)
    with pytest.raises(ValueError):
        wrap_legacy_detections(dataset, detections, tmp_path / "wrapped.jsonl", detector=row["detector"],
            synthetic=True, time_unit="seconds")
    assert not (tmp_path / "wrapped.jsonl").exists()


@pytest.mark.parametrize("bad", ["receipt_source", "receipt_model", "receipt_code", "receipt_params", "missing_source_field"])
def test_receipt_and_optional_source_binding_cannot_lose_declared_identity(source, tmp_path, bad):
    dataset, _, detections, row = source
    if bad == "missing_source_field":
        row["source_binding"] = {"artifact_sha256": "a" * 64}
        write_row(detections, row)
        with pytest.raises(ValueError):
            analyze_structure(dataset, detections, tmp_path / "structure.json")
    else:
        write_row(detections, legacy_row(source))
        receipt = {"output_sha256": file_sha(detections)}
        if bad == "receipt_source": receipt["output_sha256"] = "b" * 64
        elif bad == "receipt_model": receipt["model"] = {"checkpoint_sha256": "b" * 64}
        elif bad == "receipt_code": receipt["runtime"] = {"beatthis_inference_sha256": "b" * 64}
        else: receipt["params"] = {}
        path = tmp_path / "receipt.json"
        write_json(path, receipt)
        with pytest.raises(ValueError):
            wrap_legacy_detections(dataset, detections, tmp_path / "wrapped.jsonl", detector=row["detector"],
                synthetic=True, time_unit="seconds", run_receipt_path=path)


def test_legacy_wrapper_requires_explicit_synthetic_and_time_units(source, tmp_path):
    dataset, _, detections, row = source
    write_row(detections, legacy_row(source))
    for synthetic, units in ((None, "seconds"), (0, "seconds"), (True, "samples")):
        with pytest.raises(ValueError):
            wrap_legacy_detections(dataset, detections, tmp_path / "wrapped.jsonl", detector=row["detector"],
                synthetic=synthetic, time_unit=units)


def test_wrapper_cli_is_executable_and_does_not_rerun_the_detector(source, tmp_path):
    dataset, _, detections, row = source
    write_row(detections, legacy_row(source))
    identity = tmp_path / "detector.json"
    write_json(identity, row["detector"])
    result = subprocess.run([sys.executable, "-m", "choreobench.music_structure", "--wrap-legacy",
        "--dataset", str(dataset), "--detections", str(detections), "--out", str(tmp_path / "wrapped.jsonl"),
        "--detector-identity", str(identity), "--synthetic", "true", "--legacy-time-unit", "seconds"],
        capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["schema"] == DETECTIONS_SCHEMA and report["inference_called"] is False
    assert report["records"] == 1


def test_real_declared_beatthis_legacy_metadata_can_be_wrapped_without_discarding_audit_binding(source, tmp_path):
    dataset, _, detections, _ = source
    detector = dict(name="beat_this", version="0.1.0", model_sha256="b" * 64,
                    code_sha256="a" * 64, parameters={"device": "cpu", "dbn": False})
    old = dict(legacy_row(source), schema="beatthis_audio_detections_v1", sample_rate=16000,
        audio_frames=16000 * 65, channels=1, duration_seconds=65.0, model_sha256=detector["model_sha256"],
        library_versions={"beat-this": "0.1.0"}, params=detector["parameters"], helper_sha256="c" * 64,
        inference_called=True, model_prediction_returned=True)
    write_row(detections, old)
    receipt_path = tmp_path / "receipt.json"
    write_json(receipt_path, dict(output_sha256=file_sha(detections), model={"checkpoint_sha256": "b" * 64},
        runtime={"beatthis_inference_sha256": "a" * 64}, params=detector["parameters"]))
    result = wrap_legacy_detections(dataset, detections, tmp_path / "wrapped.jsonl", detector=detector,
        synthetic=False, time_unit="seconds", run_receipt_path=receipt_path)
    assert result[0]["synthetic"] is False
    assert result[0]["source_binding"]["artifact_sha256"] == file_sha(detections)
    assert result[0]["source_binding"]["run_receipt_sha256"] == file_sha(receipt_path)
    output = analyze_structure(dataset, tmp_path / "wrapped.jsonl", tmp_path / "structure.json")
    assert output["songs"][0]["synthetic"] is True  # The source dataset is an explicitly synthetic waveform fixture.
    assert output["songs"][0]["music"]["source"].startswith("beat_this@0.1.0+")


def test_mixed_split_analysis_selects_public_output_but_keeps_original_manifest_and_full_input_binding(source, tmp_path):
    dataset, song, detections, row = source
    value = read_json(dataset)
    other = copy.deepcopy(song)
    other.update(song_id="training", split="train")
    audio = tmp_path / "train.wav"
    payload = bytearray((tmp_path / "source.wav").read_bytes())
    payload[-1] ^= 1  # Different valid PCM, avoiding cross-split byte-identity leakage in the fixture.
    audio.write_bytes(payload)
    other["audio"].update(path="train.wav", sha256=file_sha(audio))
    value["songs"].append(other)
    write_json(dataset, value, overwrite=True)
    other_row = copy.deepcopy(row)
    other_row.update(id="training", audio_sha256=other["audio"]["sha256"])
    detections.write_text(json.dumps(row) + "\n" + json.dumps(other_row) + "\n", encoding="utf-8")
    manifest_sha = file_sha(dataset)
    result = analyze_structure(dataset, detections, tmp_path / "dev.json", splits=["dev"])
    assert result["dataset_sha256"] == manifest_sha
    assert [r["song_id"] for r in result["songs"]] == [song["song_id"]]
    effective, _, _ = load_public_structure(tmp_path / "dev.json", manifest_sha, [song])
    assert len(effective) == 1
    cli = subprocess.run([sys.executable, "-m", "choreobench.music_structure", "--dataset", str(dataset),
        "--detections", str(detections), "--out", str(tmp_path / "cli-dev.json"), "--split", "dev"],
        capture_output=True, text=True, check=True)
    assert json.loads(cli.stdout)["songs"] == 1
    all_songs = analyze_structure(dataset, detections, tmp_path / "all.json")
    assert [r["song_id"] for r in all_songs["songs"]] == [song["song_id"], "training"]
    other_row["status"] = "failed"
    detections.write_text(json.dumps(row) + "\n" + json.dumps(other_row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        analyze_structure(dataset, detections, tmp_path / "not-published.json", splits=["dev"])
    assert file_sha(dataset) == manifest_sha
    assert not (tmp_path / "not-published.json").exists()


@pytest.mark.parametrize("splits", [[], ["unknown"], ["dev", "dev"], [True], ["test"], ("dev",)])
def test_analysis_rejects_empty_unknown_repeated_or_unmatched_split_selection(source, tmp_path, splits):
    dataset, _, detections, _ = source
    with pytest.raises(ValueError):
        analyze_structure(dataset, detections, tmp_path / "structure.json", splits=splits)
