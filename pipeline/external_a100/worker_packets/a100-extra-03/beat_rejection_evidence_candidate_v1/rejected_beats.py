"""CPU-only refusal evidence for a future caller; not a production entry point."""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re

import numpy as np

ARRAY_CAP = 4 * 2**20
JSON_CAP = 2**20
FLAGS = dict(diagnostic_not_production=True, semantic_approval=False,
             quality_approved=False, training_eligible=False, needs_visual_revision=True)


def rejection_predicates(beats, downbeats, audio_duration):
    predicates = []
    for name, values in (("beats", beats), ("downbeats", downbeats)):
        if values.ndim != 1:
            predicates.append(name + ".not_one_dimensional")
        if not np.isfinite(values).all():
            predicates.append(name + ".not_finite")
        if np.any(values < 0):
            predicates.append(name + ".negative")
        if np.any(values > audio_duration + 0.02):
            predicates.append(name + ".outside_audio_duration_plus_0p02")
        if values.ndim > 0 and np.any(np.diff(values) <= 0):
            predicates.append(name + ".not_strictly_increasing")
    if beats.ndim == downbeats.ndim == 1:
        if any(not np.isclose(beat, beats, atol=1e-6, rtol=0).any() for beat in downbeats):
            predicates.append("downbeats.not_in_beats_atol_1e_minus6_rtol0")
    return predicates


def validate_and_record(beats, downbeats, audio_duration, evidence_dir, context):
    """Valid outputs keep the core float conversion; rejected numeric arrays are write-once."""
    duration = float(audio_duration)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Actual audio duration must be finite and positive")
    raw_beats, raw_downbeats = np.asarray(beats), np.asarray(downbeats)
    if any(v.dtype.kind not in "biufc" for v in (raw_beats, raw_downbeats)):
        raise TypeError("Refusal evidence requires non-object numeric model arrays")
    converted = tuple(np.asarray(v, dtype=float) for v in (raw_beats, raw_downbeats))
    predicates = rejection_predicates(*converted, duration)
    if not predicates:
        return converted
    if raw_beats.nbytes + raw_downbeats.nbytes > ARRAY_CAP:
        raise ValueError("Refusal array evidence exceeds the bounded 4MiB budget; source remains failed")
    required = {"source_row", "worker", "gpu", "assignment_index", "assignment_sha256",
                "algorithm_sha256", "model_sha256", "device", "float16", "dbn"}
    if set(context) != required:
        raise ValueError("Refusal evidence caller binding fields differ")
    row = context["source_row"]
    if not isinstance(row, dict) or not all(isinstance(row.get(k), str) and row[k] for k in ("id", "split", "audio_path")):
        raise ValueError("Refusal evidence lacks the original source binding")
    if type(context["gpu"]) is not int or context["gpu"] not in range(4) or type(context["assignment_index"]) is not int or context["assignment_index"] < 0 or context["assignment_index"] % 4 != context["gpu"]:
        raise ValueError("Refusal evidence assignment row/GPU binding differs")
    if context["worker"] not in tuple("a100-extra-%02d" % n for n in range(3, 7)) or context["device"] != "cuda:0" or context["float16"] is not False or context["dbn"] is not False:
        raise ValueError("Refusal evidence worker/model configuration differs")
    if any(not isinstance(context[k], str) or re.fullmatch(r"[0-9a-f]{64}", context[k]) is None for k in ("assignment_sha256", "algorithm_sha256", "model_sha256")):
        raise ValueError("Refusal evidence model/algorithm/assignment SHA binding invalid")
    directory = Path(evidence_dir)
    if not directory.is_absolute() or directory.resolve(strict=True) != directory:
        raise ValueError("Refusal output must be an existing canonical caller-authorized directory")
    buffer = io.BytesIO()
    np.savez(buffer, beats=raw_beats, downbeats=raw_downbeats)
    payload = buffer.getvalue()
    if len(payload) > ARRAY_CAP + 65536:
        raise ValueError("Refusal NPZ evidence budget exceeded")
    digest = hashlib.sha256(payload).hexdigest()
    record = dict(schema="ms_beatthis_rejection_evidence_v1",
                  created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  status="failed_sealed_no_retry", error="Invalid actual BeatThis beat/downbeat output",
                  audio_duration_seconds=duration, predicates=predicates,
                  numeric_conversion="same np.asarray(dtype=float) as frozen core; raw arrays unchanged",
                  arrays={name: dict(dtype=str(v.dtype), shape=list(v.shape))
                          for name, v in (("beats", raw_beats), ("downbeats", raw_downbeats))},
                  archive=dict(name="rejected_beats.npz", bytes=len(payload), sha256=digest),
                  source_reexecuted=False, validation_relaxed=False, **context, **FLAGS)
    metadata = (json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + chr(10)).encode()
    if len(metadata) > JSON_CAP:
        raise ValueError("Refusal evidence metadata budget exceeded")
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name, raw in (("rejected_beats.npz", payload), ("rejection.json", metadata)):
            output = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444, dir_fd=fd)
            with os.fdopen(output, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.fsync(fd)
            readfd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
            with os.fdopen(readfd, "rb") as stream:
                before = os.fstat(stream.fileno())
                actual = stream.read(len(raw) + 1)
                if before.st_size != len(raw) or actual != raw or os.fstat(stream.fileno()).st_size != len(raw):
                    raise ValueError("Refusal evidence opened-FD readback differs; retain files")
    finally:
        os.close(fd)
    raise ValueError("Invalid actual BeatThis beat/downbeat output")
