import copy
import wave

import pytest

from choreobench.dataset import assemble_dataset, import_structure, load_dataset, public_song, release_report
from choreobench.fixtures import create_demo
from choreobench.io import object_sha, read_json, write_json
from choreobench.schema import ValidationError, validate_dataset, validate_plan


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / "demo"
    create_demo(root)
    return root


def test_fixture_is_not_a_benchmark_release(fixture):
    data = load_dataset(fixture / "dataset.json")
    report = release_report(data)
    assert report["eligible_by_declared_contract"] is False
    assert len(report["blockers"]) >= 3
    model_input = public_song(data["songs"][0])
    assert "split" not in model_input and "audio" not in model_input
    assert "artist_group_id" not in model_input


def test_actual_audio_hash_and_clock_required(fixture):
    path = fixture / "dataset.json"
    data = read_json(path)
    data["songs"][0]["audio"]["num_samples"] -= 1
    data["songs"][0]["music"]["sections"][-1]["end"] -= 1
    write_json(path, data, overwrite=True)
    with pytest.raises(ValueError, match="clock"):
        load_dataset(path)


def test_cross_split_song_artist_duplicate_and_audio_leaks(fixture):
    original = read_json(fixture / "dataset.json")
    for shared in ("song_group_id", "artist_group_id", "near_duplicate_group_id", "audio_sha256"):
        data = copy.deepcopy(original)
        twin = copy.deepcopy(data["songs"][0])
        twin.update(song_id="twin", split="test", song_group_id="other-song", artist_group_id="other-artist", near_duplicate_group_id="other-near")
        twin["audio"]["sha256"] = "a" * 64
        if shared == "audio_sha256":
            twin["audio"]["sha256"] = data["songs"][0]["audio"]["sha256"]
        else:
            twin[shared] = data["songs"][0][shared]
        data["songs"].append(twin)
        with pytest.raises(ValidationError, match="Cross-split"):
            validate_dataset(data)


def test_bool_and_float_sample_times_rejected(fixture):
    plan = read_json(fixture / "source_plan.json")
    for bad in (True, 0.0, float("nan"), -1):
        changed = copy.deepcopy(plan)
        changed["events"][0]["start"] = bad
        with pytest.raises(ValidationError):
            validate_plan(changed)


def test_multitrack_valid_but_wholebody_overlap_invalid(fixture):
    plan = read_json(fixture / "source_plan.json")
    validate_plan(plan)
    changed = copy.deepcopy(plan)
    changed["events"][0]["track"] = "body"
    changed["events"].sort(key=lambda x: (x["start"], x["track"], x["event_id"]))
    with pytest.raises(ValidationError, match="Overlapping"):
        validate_plan(changed)


def test_plan_clock_mismatch_and_unknown_motif(fixture):
    plan = read_json(fixture / "source_plan.json")
    song = read_json(fixture / "dataset.json")["songs"][0]
    plan["sample_rate"] *= 2
    with pytest.raises(ValidationError, match="clock"):
        validate_plan(plan, song)
    plan["sample_rate"] //= 2
    plan["events"][0]["motif_id"] = "missing"
    with pytest.raises(ValidationError, match="motif"):
        validate_plan(plan)


def test_strict_json_and_exclusive_publication(tmp_path):
    path = tmp_path / "data.json"
    for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
        path.write_text(raw)
        with pytest.raises(ValueError):
            read_json(path)
    with pytest.raises(ValueError):
        write_json(tmp_path / "invalid.json", {"a": float("nan")})
    write_json(path, {"new": 1}, overwrite=True)
    with pytest.raises(FileExistsError):
        write_json(path, {"new": 2})
    assert read_json(path) == {"new": 1}


def test_structure_import_requires_full_partition_and_binding(fixture):
    data = read_json(fixture / "dataset.json")
    song = data["songs"][0]
    structure = fixture / "tool.json"
    write_json(structure, {"beats": [0, .5], "downbeats": [0], "segments": [{"start": 0, "end": 12, "label": "verse"}]})
    with pytest.raises(ValidationError, match="SHA256"):
        import_structure(fixture / "dataset.json", "demo-song", structure, fixture / "bad.json", audio_sha256="f" * 64)
    out = fixture / "nested" / "dataset.json"
    imported = import_structure(fixture / "dataset.json", "demo-song", structure, out, audio_sha256=song["audio"]["sha256"])
    assert load_dataset(out)["songs"][0]["music"]["beats"] == [0, 8000]
    assert imported["songs"][0]["audio"]["path"] == "../synthetic.wav"
    # Importer must not silently fill missing intervals or invent a repeat group.
    write_json(structure, {"beats": [], "downbeats": [], "segments": [{"start": 1, "end": 12, "label": "verse"}]}, overwrite=True)
    with pytest.raises(ValidationError, match="partition"):
        import_structure(fixture / "dataset.json", "demo-song", structure, fixture / "gap.json", audio_sha256=song["audio"]["sha256"])


def test_prepare_audio_downmix_resample_and_durable_publication(tmp_path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("scipy")
    from choreobench.dataset import prepare_audio, inspect_wav
    source = tmp_path / "stereo.wav"
    # Exactly 6 seconds, stereo 24 kHz; test the optional preparation path.
    samples = np.zeros((24000 * 6, 2), dtype=np.float32)
    samples[:, 0] = .2
    samples[:, 1] = -.1
    sf.write(source, samples, 24000, subtype="FLOAT")
    out = tmp_path / "canonical.wav"
    manifest = tmp_path / "song.json"
    prepared = prepare_audio(source, out, manifest, song_id="prepared", sample_rate=16000)
    actual = inspect_wav(out)
    assert actual["num_samples"] == 16000 * 6
    assert actual["sample_rate"] == 16000
    assert actual["sha256"] == prepared["audio"]["sha256"]
    receipt = read_json(tmp_path / "song.audio_receipt.json")
    assert receipt["source_channels"] == 2
    assert receipt["source_cropped"] is False
    with pytest.raises(FileExistsError):
        prepare_audio(source, out, manifest, song_id="prepared")


def test_wav_validation_accepts_different_path_stat_metadata(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    from choreobench.dataset import inspect_wav
    from choreobench.io import file_sha
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x00\x00" * 80)
    original_stat = Path.stat
    def different_path_identity(path, *args, **kwargs):
        stat = original_stat(path, *args, **kwargs)
        if path != audio:
            return stat
        return SimpleNamespace(st_dev=stat.st_dev + 1, st_ino=stat.st_ino + 1,
                               st_size=stat.st_size, st_mtime_ns=stat.st_mtime_ns,
                               st_ctime_ns=stat.st_ctime_ns + 1)
    expected_sha = file_sha(audio)
    monkeypatch.setattr(Path, "stat", different_path_identity)
    assert inspect_wav(audio) == {"sample_rate": 16000, "num_samples": 80,
                                  "sha256": expected_sha}


@pytest.mark.parametrize("change_call,field", [(2, "st_mtime_ns"), (3, "st_ino")])
def test_wav_validation_still_rejects_mutation_or_replacement(tmp_path, monkeypatch, change_call, field):
    from types import SimpleNamespace
    from choreobench import dataset
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x00\x00" * 80)
    original_fstat = dataset.os.fstat
    calls = 0
    def changing_identity(fd):
        nonlocal calls
        calls += 1
        stat = original_fstat(fd)
        values = {key: getattr(stat, key) for key in
                  ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}
        if calls == change_call:
            values[field] += 1
        return SimpleNamespace(**values)
    monkeypatch.setattr(dataset.os, "fstat", changing_identity)
    with pytest.raises(ValidationError, match="changed during validation"):
        dataset.inspect_wav(audio)
