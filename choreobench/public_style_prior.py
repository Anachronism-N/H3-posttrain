"""Read-only public JEV probabilities, separate from the exact song/music schema.

v1 accepts only the published model identity below. Windows use global, half-open
integer samples and complete 5.0625-second observations (nearest sample, half up).
Window hashes are declarations, not verified waveform identities or genre truth.
No reference captions/private labels, inference, calibration or fallback is added.
"""
import copy
import math
from pathlib import Path
import re

from .io import file_sha, read_json_snapshot
from .schema import identifier, integer, keys

SCHEMA = "choreobench.public_style_prior.v1"
CLASS_ORDER = (
    "pop / casual", "hip-hop", "jazz funk / heels", "chinese classical",
    "chinese folk", "contemporary", "k-pop",
)
MODEL_IDENTITY = dict(
    archive_sha256="e59465a82b2411ff89981445848430b38bebbce6af7f583fd3fa19692d0c4d75",
    architecture="frozen MuQ + 3 fresh Jev heads",
    class_order=list(CLASS_ORDER), calibrated=True,
    calibration_scope="Temperature fit on held-out video-VLM labels; not human music-suitability calibration",
    member_head_sha256=[
        "1323e17e157b26f3557c4d60e629c69910b1c266b256892560f4483dfbc8d9b6",
        "c68dd809258eb46ec7322ba0c680665fc1423a035bc0ade898ac41104ac4483b",
        "3ca76a8a81d950e2fd8a4e561b2d6ce10b55ee35dbf18131f640f8598b6dec75",
    ],
    encoder_weights_sha256="273febab2be02872c37d2c37e48a9d6c52c1c9392f3eeeabd498efa281ccb7a6",
    pseudolabel_caveat="Model P predictions trained on video VLM labels; probabilities are fallible "
                      "recommendations, not audited true or unique dance genres.",
)


def _sha(value, where):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError(where + " must be lowercase SHA256")


def _validate(document, dataset_sha, songs):
    keys(document, ("schema", "dataset_sha256", "model_identity", "songs"), where="public style prior")
    _sha(document["dataset_sha256"], "style prior dataset_sha256")
    if document["schema"] != SCHEMA or document["dataset_sha256"] != dataset_sha:
        raise ValueError("Public style prior must bind the exact source dataset")
    identity = document["model_identity"]
    keys(identity, MODEL_IDENTITY, where="public style prior model identity")
    if type(identity["calibrated"]) is not bool or identity != MODEL_IDENTITY:
        raise ValueError("Public style prior published model identity/class order/caveat differs")
    if not isinstance(document["songs"], list):
        raise ValueError("Public style prior songs must be a list")
    originals = {song["song_id"]: song for song in songs}
    if len(originals) != len(songs):
        raise ValueError("Duplicate source song_id")
    seen = set()
    for row in document["songs"]:
        keys(row, ("song_id", "audio_sha256", "sample_rate", "num_samples", "windows"),
             where="public style prior song")
        identifier(row["song_id"], "style prior song_id")
        if row["song_id"] in seen or row["song_id"] not in originals:
            raise ValueError("Duplicate/unknown public style prior song_id")
        seen.add(row["song_id"])
        _sha(row["audio_sha256"], "style prior audio_sha256")
        integer(row["sample_rate"], "style prior sample_rate", 1)
        integer(row["num_samples"], "style prior num_samples", 1)
        audio = originals[row["song_id"]]["audio"]
        if (row["audio_sha256"] != audio["sha256"] or any(row[k] != audio[k]
                for k in ("sample_rate", "num_samples"))):
            raise ValueError("Public style prior audio hash/clock differs from source audio")
        if not isinstance(row["windows"], list):
            raise ValueError("Public style prior windows must be a list; [] explicitly means unavailable")
        intervals = set()
        length = (81 * row["sample_rate"] + 8) // 16
        for window in row["windows"]:
            keys(window, ("start", "end", "audio_sha256", "probabilities"), where="style prior window")
            integer(window["start"], "style prior window.start")
            integer(window["end"], "style prior window.end", 1)
            interval = (window["start"], window["end"])
            if (not 0 <= interval[0] < interval[1] <= row["num_samples"]
                    or interval[1] - interval[0] != length):
                raise ValueError("Style prior requires complete in-bounds 5.0625s audio observations")
            if interval in intervals:
                raise ValueError("Duplicate style prior window interval")
            intervals.add(interval)
            _sha(window["audio_sha256"], "style prior window.audio_sha256")
            probabilities = window["probabilities"]
            if (not isinstance(probabilities, list) or len(probabilities) != len(CLASS_ORDER)
                    or any(type(p) not in (int, float) or p < 0 or p > 1 + 1e-6 or not math.isfinite(p) for p in probabilities)
                    or abs(math.fsum(probabilities) - 1) > 1e-6):
                raise ValueError("Style prior probabilities require seven finite nonnegative ordered values summing to 1 +/- 1e-6")
    if seen != set(originals):
        raise ValueError("Public style prior must contain exactly the source dataset song set")


class PublicStylePrior:
    def __init__(self, path, document, digest):
        self.path = Path(path).resolve()
        self.digest = digest
        self.document = copy.deepcopy(document)
        self._songs = {row["song_id"]: row for row in self.document["songs"]}

    def binding(self):
        return dict(public_style_prior_sha256=self.digest,
                    public_style_prior_snapshot=copy.deepcopy(self.document))

    def assert_unchanged(self):
        if file_sha(self.path) != self.digest:
            raise ValueError("Public style prior source bytes changed; start a new run")

    def signal(self, song_id, unit=None):
        """Equal-window global mean or overlap-sample weighted unit mean; no imputation.

        Overlapping windows each contribute their declared vector, not independent
        observations of genre truth. Coverage is the union, not summed weights.
        Input probabilities are not renormalized; ties use canonical class order.
        """
        row = self._songs[song_id]
        start, end = (0, row["num_samples"]) if unit is None else (unit["start"], unit["end"])
        integer(start, "style prior scope.start")
        integer(end, "style prior scope.end", 1)
        if not 0 <= start < end <= row["num_samples"]:
            raise ValueError("Style prior scope escapes source audio")
        selected = []
        for window in sorted(row["windows"], key=lambda w: (w["start"], w["end"])):
            left, right = max(start, window["start"]), min(end, window["end"])
            if left < right:
                selected.append((window, left, right, 1 if unit is None else right - left))
        total = sum(weight for _, _, _, weight in selected)
        probabilities = ([math.fsum(w["probabilities"][i] * (weight / total)
                          for w, _, _, weight in selected) for i in range(len(CLASS_ORDER))] if total else None)
        top3 = ([] if probabilities is None else [dict(label=CLASS_ORDER[i], probability=probabilities[i])
                for i in sorted(range(len(CLASS_ORDER)), key=lambda i: (-probabilities[i], i))[:3]])
        coverage, cursor = 0, start
        for _, left, right, _ in selected:
            coverage += max(0, right - max(left, cursor))
            cursor = max(cursor, right)
        return dict(schema=SCHEMA, sidecar_sha256=self.digest, dataset_sha256=self.document["dataset_sha256"],
            model_identity=copy.deepcopy(self.document["model_identity"]), song_id=song_id,
            audio_sha256=row["audio_sha256"], sample_rate=row["sample_rate"], num_samples=row["num_samples"],
            scope=dict(kind="global" if unit is None else "unit", start=start, end=end),
            aggregation=dict(method="equal_window_mean" if unit is None else "overlap_sample_weighted_window_mean",
                window_count=len(selected), weight_sum=total, coverage_samples=coverage,
                missing_policy="No imputation or global fallback for missing/nonoverlapping windows",
                top3_tie_break="canonical class order", window_audio_hashes_verified=False),
            status="available" if total else "missing_windows" if not row["windows"] else "no_overlapping_windows",
            probabilities=probabilities, top3=top3,
            usage="Optional fallible recommendation only, never a hard style/genre constraint; do not override the user brief.")


def load_public_style_prior(path, dataset_sha, songs):
    """Validate against the entire bound source dataset, including unselected songs."""
    if path is None:
        return None
    document, digest = read_json_snapshot(Path(path))
    _validate(document, dataset_sha, songs)
    return PublicStylePrior(path, document, digest)
