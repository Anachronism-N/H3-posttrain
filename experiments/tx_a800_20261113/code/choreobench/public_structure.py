"""Strict public music sidecars, independently bound to the unchanged candidate dataset."""
import copy
from pathlib import Path

from .io import object_sha, read_json_snapshot
from .schema import keys, validate_song


def load_public_structure(path, dataset_sha, songs):
    """Return effective songs and a snapshot; never loosen candidate bank matching."""
    if path is None:
        return copy.deepcopy(songs), None, None
    document, digest = read_json_snapshot(Path(path))
    keys(document, ("schema", "dataset_sha256", "songs"), where="public structure")
    if document["schema"] != "choreobench.public_structure.v1" or document["dataset_sha256"] != dataset_sha:
        raise ValueError("Public structure must bind the exact source dataset")
    if not isinstance(document["songs"], list):
        raise ValueError("Public structure songs must be a list")
    by_id = {}
    for row in document["songs"]:
        keys(row, ("song_id", "audio_sha256", "sample_rate", "num_samples", "music", "synthetic"),
             where="public structure song")
        if not isinstance(row["song_id"], str) or row["song_id"] in by_id or type(row["synthetic"]) is not bool:
            raise ValueError("Duplicate/invalid public structure song")
        by_id[row["song_id"]] = row
    if set(by_id) != {s["song_id"] for s in songs}:
        raise ValueError("Public structure must contain exactly the selected source songs")
    effective = []
    for song in songs:
        row = by_id[song["song_id"]]
        audio = song["audio"]
        if (row["audio_sha256"] != audio["sha256"] or type(row["sample_rate"]) is not int
                or type(row["num_samples"]) is not int or row["sample_rate"] != audio["sample_rate"]
                or row["num_samples"] != audio["num_samples"]):
            raise ValueError("Public structure audio hash/clock differs from the original candidate input")
        result = copy.deepcopy(song)
        result["music"] = copy.deepcopy(row["music"])
        validate_song(result)
        effective.append(result)
    return effective, document, digest
