"""Portable manifests, waveform verification, and explicit music-tool imports."""
import copy
import hashlib
import math
import os
import tempfile
import wave
from pathlib import Path

from .io import file_sha, object_sha, read_json, read_json_snapshot, write_json
from .schema import ValidationError, identifier, validate_dataset, validate_song


def resolve_audio(song, dataset_path):
    path = Path(song["audio"]["path"]).expanduser()
    return path.resolve() if path.is_absolute() else (Path(dataset_path).resolve().parent / path).resolve()


def inspect_wav(path):
    """Core/native baseline contract: decoded, mono, signed 16-bit PCM WAV."""
    path = Path(path)
    def identity(stat):
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    try:
        with path.open("rb") as raw:
            before = identity(os.fstat(raw.fileno()))
            with wave.open(raw, "rb") as stream:
                if (stream.getnchannels(), stream.getsampwidth(), stream.getcomptype()) != (1, 2, "NONE"):
                    raise ValidationError("Canonical audio must be mono PCM16 WAV; use prepare-audio")
                frames, rate = stream.getnframes(), stream.getframerate()
                read_frames = 0
                while True:
                    block = stream.readframes(65536)
                    if not block:
                        break
                    if len(block) % 2:
                        raise ValidationError("WAV has a partial sample")
                    read_frames += len(block) // 2
                if frames != read_frames or frames <= 0 or rate <= 0:
                    raise ValidationError("WAV payload is truncated or empty")
            raw.seek(0)
            digest = hashlib.sha256()
            for block in iter(lambda: raw.read(8 * 1024 * 1024), b""):
                digest.update(block)
            if before != identity(os.fstat(raw.fileno())) or before != identity(path.stat()):
                raise ValidationError("WAV changed during validation")
    except (wave.Error, EOFError) as error:
        raise ValidationError("Invalid PCM WAV: " + str(path)) from error
    return {"sample_rate": rate, "num_samples": frames, "sha256": digest.hexdigest()}


def load_dataset(path, verify_audio=True):
    dataset = validate_dataset(read_json(path))
    if verify_audio:
        verify_dataset_audio(dataset, path)
    return dataset


def verify_dataset_audio(dataset, path):
    validate_dataset(dataset)
    for song in dataset["songs"]:
        actual = inspect_wav(resolve_audio(song, path))
        if any(actual[key] != song["audio"][key] for key in actual):
            raise ValidationError("Audio identity/clock mismatch for " + song["song_id"])
    return dataset


def public_song(song):
    """Exact model-facing whitelist. No split/group IDs, private labels or paths."""
    validate_song(song)
    return {"song_id": song["song_id"], "brief": song["brief"],
            "sample_rate": song["audio"]["sample_rate"], "num_samples": song["audio"]["num_samples"],
            "music": copy.deepcopy(song["music"])}


def release_report(dataset, track="full_song"):
    validate_dataset(dataset)
    if track not in ("local", "full_song"):
        raise ValidationError("track must be local/full_song")
    blockers = []
    if dataset["synthetic"]:
        blockers.append("Synthetic fixtures are excluded from scientific benchmark results")
    if dataset["grouping"]["status"] != "reviewed":
        blockers.append("Song/artist/near-duplicate grouping lacks a documented external review")
    for song in dataset["songs"]:
        missing = [name for name in ("song_group_id", "artist_group_id", "near_duplicate_group_id") if song[name] is None]
        if missing:
            blockers.append(song["song_id"] + ": missing " + ",".join(missing))
        seconds = song["audio"]["num_samples"] / song["audio"]["sample_rate"]
        if track == "full_song" and not 180 <= seconds <= 300:
            blockers.append(song["song_id"] + ": full_song core track requires 180..300 seconds")
        if track == "local" and abs(seconds - 5.0625) > 1 / song["audio"]["sample_rate"]:
            blockers.append(song["song_id"] + ": native local track requires 5.0625 seconds")
    return {"schema": "choreobench.release_check.v1", "dataset_sha256": object_sha(dataset), "track": track,
            "eligible_by_declared_contract": not blockers, "blockers": blockers,
            "limitations": ["Grouping declarations are traceability, not proof of semantic deduplication",
                            "This check does not certify rights, label quality, sample size, or model performance"]}


def _portable_path(path, manifest_path):
    try:
        return Path(os.path.relpath(str(Path(path).resolve()), str(Path(manifest_path).resolve().parent))).as_posix()
    except ValueError:  # Different Windows drives cannot have relative paths.
        return str(Path(path).resolve())


def relocate_dataset(dataset, source_path, target_path):
    result = copy.deepcopy(dataset)
    for original, song in zip(dataset["songs"], result["songs"]):
        song["audio"]["path"] = _portable_path(resolve_audio(original, source_path), target_path)
    return result


def prepare_audio(source, out_wav, out_song, *, song_id, split="dev", sample_rate=16000,
                  song_group_id=None, artist_group_id=None, near_duplicate_group_id=None, brief=""):
    """Explicit conversion only. Preserves timing, records downmix/resampling receipt."""
    identifier(song_id, "song_id")
    if type(sample_rate) is not int or not 8000 <= sample_rate <= 192000:
        raise ValidationError("sample_rate must be an integer in 8000..192000")
    source, out_wav, out_song = Path(source).resolve(), Path(out_wav).resolve(), Path(out_song).resolve()
    receipt_path = out_song.with_name(out_song.stem + ".audio_receipt.json")
    destinations = (out_wav, out_song, receipt_path)
    if len(set(destinations)) != 3 or source in destinations:
        raise ValidationError("Input and all output paths must be distinct")
    for path in destinations:
        if path.exists():
            raise FileExistsError(str(path))
    prototype = {"song_id": song_id, "split": split, "song_group_id": song_group_id,
                 "artist_group_id": artist_group_id, "near_duplicate_group_id": near_duplicate_group_id,
                 "brief": brief, "audio": {"path": "pending.wav", "sha256": "0" * 64, "sample_rate": sample_rate, "num_samples": 1},
                 "music": {"source": "unavailable", "beats": [], "downbeats": [], "sections": [
                     {"section_id": "s000", "start": 0, "end": 1, "label": "unknown", "repeat_group": None}]}}
    validate_song(prototype)
    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
    except ImportError as error:
        raise RuntimeError("prepare-audio requires requirements-audio.txt (numpy, scipy, soundfile)") from error
    source_sha = file_sha(source)
    samples, source_rate = sf.read(str(source), dtype="float64", always_2d=True)
    if not samples.size or not np.isfinite(samples).all():
        raise ValidationError("Source waveform is empty or nonfinite")
    if file_sha(source) != source_sha:
        raise ValidationError("Source changed during decoding")
    original_frames, original_channels = samples.shape
    mono = samples.mean(axis=1)
    if source_rate != sample_rate:
        divisor = math.gcd(source_rate, sample_rate)
        mono = resample_poly(mono, sample_rate // divisor, source_rate // divisor)
    if float(np.abs(mono).max()) > 1.0:
        raise ValidationError("PCM16 conversion would clip; explicitly normalize source and record that preprocessing first")
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=out_wav.parent, suffix=".wav", delete=False) as stream:
            temporary = Path(stream.name)
        sf.write(str(temporary), mono, sample_rate, subtype="PCM_16", format="WAV")
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.link(str(temporary), str(out_wav))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    audio = inspect_wav(out_wav)
    song = copy.deepcopy(prototype)
    song["audio"] = dict(audio, path=_portable_path(out_wav, out_song))
    song["music"]["sections"][0]["end"] = audio["num_samples"]
    validate_song(song)
    write_json(out_song, song)
    receipt = {"schema": "choreobench.audio_preparation.v1", "source_sha256": source_sha,
               "source_sample_rate": source_rate, "source_frames": original_frames,
               "source_channels": original_channels, "canonical": song["audio"],
               "operations": ["mean-channel-downmix", "scipy-resample-poly-if-needed", "PCM16-quantization"],
               "gain_normalization": False, "source_cropped": False}
    write_json(receipt_path, receipt)
    return song


def assemble_dataset(song_paths, out, *, dataset_id, synthetic=False):
    songs = []
    for path in song_paths:
        song = validate_song(read_json(path))
        song = copy.deepcopy(song)
        song["audio"]["path"] = _portable_path(resolve_audio(song, path), out)
        songs.append(song)
    dataset = {"schema": "choreobench.dataset.v1", "dataset_id": dataset_id, "synthetic": synthetic,
               "grouping": {"status": "unverified", "evidence": ""}, "songs": songs}
    validate_dataset(dataset)
    write_json(out, dataset)
    return dataset


def import_structure(dataset_path, song_id, structure_path, out, *, audio_sha256, format="allin1"):
    manifest_sha = file_sha(dataset_path)
    dataset = load_dataset(dataset_path)
    if file_sha(dataset_path) != manifest_sha:
        raise ValidationError("Dataset changed during structure import")
    selected = [song for song in dataset["songs"] if song["song_id"] == song_id]
    if len(selected) != 1:
        raise ValidationError("song_id must identify exactly one dataset song")
    song = selected[0]
    if audio_sha256 != song["audio"]["sha256"]:
        raise ValidationError("Structure must explicitly bind the canonical audio SHA256")
    raw, structure_sha = read_json_snapshot(structure_path)
    if format == "samples":
        music = copy.deepcopy(raw)
    elif format == "allin1":
        if not isinstance(raw, dict) or not {"beats", "downbeats", "segments"}.issubset(raw):
            raise ValidationError("allin1 import requires beats/downbeats/segments")
        rate = song["audio"]["sample_rate"]
        def sample(value):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValidationError("Tool timestamps must be finite nonnegative seconds")
            return round(value * rate)
        music = {"source": "allin1:" + structure_sha,
                 "beats": [sample(t) for t in raw["beats"]], "downbeats": [sample(t) for t in raw["downbeats"]],
                 "sections": [{"section_id": "s%03d" % i, "start": sample(s["start"]), "end": sample(s["end"]),
                               "label": s["label"], "repeat_group": None} for i, s in enumerate(raw["segments"])]}
    else:
        raise ValidationError("structure format must be allin1/samples")
    song["music"] = music
    validate_dataset(dataset)
    dataset = relocate_dataset(dataset, dataset_path, out)
    receipt_path = Path(out).with_name(Path(out).stem + ".structure_receipt.json")
    if Path(out).exists() or receipt_path.exists():
        raise FileExistsError("Structure output or its receipt already exists")
    write_json(out, dataset)
    write_json(receipt_path, {
        "schema": "choreobench.structure_import.v1", "song_id": song_id, "canonical_audio_sha256": audio_sha256,
        "tool_output_sha256": structure_sha, "source_manifest_sha256": manifest_sha,
        "binding": "operator-declared canonical waveform; tool accuracy not verified"})
    return dataset
