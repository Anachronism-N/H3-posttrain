"""Synthetic waveform checks for public structure evidence, not detector quality."""
import json
import math
import struct
import wave

import pytest

from choreobench.io import file_sha, read_json, write_json
from choreobench.music_structure import analyze_structure
from choreobench.public_structure import load_public_structure


@pytest.fixture
def source(tmp_path):
    rate, count = 16000, 16000 * 65
    wav = tmp_path / "source.wav"
    pcm = b"".join(struct.pack("<h", int(3000 * math.sin(2 * math.pi * (220 if i < count // 2 else 2200) * i / rate)))
                   for i in range(count))
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
    row = dict(id="test", status="ok", audio_sha256=file_sha(wav), model_sha256="a" * 64,
               beats=list(range(1, 65)), downbeats=list(range(1, 65, 2)))
    detections.write_text(json.dumps(row) + "\n")
    return dataset, song, detections, row


def test_novelty_sidecar_preserves_clock_unknown_labels_and_source_dataset(source, tmp_path):
    pytest.importorskip("numpy"); pytest.importorskip("soundfile"); pytest.importorskip("scipy")
    dataset, song, detections, _ = source
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
