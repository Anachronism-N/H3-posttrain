"""Versioned public contracts; no learned or aesthetic scoring here."""
import math
import re

from .io import finite_json


class ValidationError(ValueError):
    pass


TRACKS = {
    "body": frozenset(("head", "torso", "left_arm", "right_arm", "left_leg", "right_leg")),
    "upper": frozenset(("head", "torso", "left_arm", "right_arm")),
    "lower": frozenset(("left_leg", "right_leg")),
}
for _track in ("head", "torso", "left_arm", "right_arm", "left_leg", "right_leg"):
    TRACKS[_track] = frozenset((_track,))


def keys(value, required, optional=(), where="object"):
    if not isinstance(value, dict):
        raise ValidationError(where + " must be an object")
    missing = set(required) - set(value)
    extra = set(value) - set(required) - set(optional)
    if missing or extra:
        raise ValidationError("%s fields: missing=%s unexpected=%s" % (where, sorted(missing), sorted(extra)))


def text(value, where, nullable=False):
    if nullable and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(where + " must be nonempty text")


def identifier(value, where):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value):
        raise ValidationError(where + " must be a safe stable identifier")


def filename_identifier(value, where):
    identifier(value, where)
    stem = value.split(".")[0].upper()
    reserved = {"CON", "PRN", "AUX", "NUL"} | {"COM%d" % i for i in range(1, 10)} | {"LPT%d" % i for i in range(1, 10)}
    if ":" in value or value.endswith(".") or stem in reserved:
        raise ValidationError(where + " must be a portable filename identifier")


def integer(value, where, low=0):
    if type(value) is not int or value < low:
        raise ValidationError(where + " must be an integer >= " + str(low))


def _interval(value, duration, where):
    integer(value.get("start"), where + ".start")
    integer(value.get("end"), where + ".end", 1)
    if not value["start"] < value["end"] <= duration:
        raise ValidationError(where + " must satisfy 0 <= start < end <= num_samples")


def _scalar_object(value, where):
    if not isinstance(value, dict):
        raise ValidationError(where + " must be an object")
    for key, item in value.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", key):
            raise ValidationError(where + " contains invalid attribute key")
        if item is not None and type(item) not in (str, int, float, bool):
            raise ValidationError(where + " values must be JSON scalars")
        if type(item) is float and not math.isfinite(item):
            raise ValidationError(where + " contains nonfinite values")


def validate_song(song):
    keys(song, ("song_id", "split", "song_group_id", "artist_group_id", "near_duplicate_group_id",
                "brief", "audio", "music"), where="song")
    filename_identifier(song["song_id"], "song.song_id")
    if song["split"] not in ("train", "dev", "test"):
        raise ValidationError("split must be train/dev/test")
    for name in ("song_group_id", "artist_group_id", "near_duplicate_group_id"):
        if song[name] is not None:
            identifier(song[name], name)
    if not isinstance(song["brief"], str):
        raise ValidationError("brief must be a string")
    audio = song["audio"]
    keys(audio, ("path", "sha256", "sample_rate", "num_samples"), where="audio")
    text(audio["path"], "audio.path")
    if not isinstance(audio["sha256"], str) or not re.fullmatch("[a-f0-9]{64}", audio["sha256"]):
        raise ValidationError("audio.sha256 must be lowercase SHA256")
    integer(audio["sample_rate"], "sample_rate", 1)
    integer(audio["num_samples"], "num_samples", 1)
    duration = audio["num_samples"]
    music = song["music"]
    keys(music, ("source", "beats", "downbeats", "sections"), where="music")
    text(music["source"], "music.source")
    for name in ("beats", "downbeats"):
        values = music[name]
        if not isinstance(values, list):
            raise ValidationError(name + " must be a list")
        for v in values:
            integer(v, name)
            if v >= duration:
                raise ValidationError(name + " outside audio")
        if values != sorted(set(values)):
            raise ValidationError(name + " must be sorted and unique")
    if not set(music["downbeats"]).issubset(music["beats"]):
        raise ValidationError("downbeats must be included in beats")
    sections = music["sections"]
    if not isinstance(sections, list) or not sections:
        raise ValidationError("sections must be a nonempty partition; use explicit unknown if unavailable")
    cursor, seen = 0, set()
    for section in sections:
        keys(section, ("section_id", "start", "end", "label", "repeat_group"), where="section")
        identifier(section["section_id"], "section_id")
        if section["section_id"] in seen:
            raise ValidationError("Duplicate section_id")
        seen.add(section["section_id"])
        _interval(section, duration, "section")
        if section["start"] != cursor:
            raise ValidationError("Sections must exactly partition the song in time order")
        cursor = section["end"]
        text(section["label"], "section.label")
        if section["repeat_group"] is not None:
            identifier(section["repeat_group"], "repeat_group")
    if cursor != duration:
        raise ValidationError("Sections do not cover the song tail")
    return song


def validate_dataset(dataset):
    keys(dataset, ("schema", "dataset_id", "synthetic", "grouping", "songs"), where="dataset")
    if dataset["schema"] != "choreobench.dataset.v1":
        raise ValidationError("Unsupported dataset schema")
    identifier(dataset["dataset_id"], "dataset_id")
    if type(dataset["synthetic"]) is not bool:
        raise ValidationError("synthetic must be boolean")
    keys(dataset["grouping"], ("status", "evidence"), where="grouping")
    if dataset["grouping"]["status"] not in ("unverified", "reviewed"):
        raise ValidationError("grouping.status must be unverified/reviewed")
    if not isinstance(dataset["grouping"]["evidence"], str):
        raise ValidationError("grouping.evidence must be text")
    if dataset["grouping"]["status"] == "reviewed":
        text(dataset["grouping"]["evidence"], "grouping.evidence")
    if not isinstance(dataset["songs"], list) or not dataset["songs"]:
        raise ValidationError("songs must be nonempty")
    seen, split_members = set(), {}
    for song in dataset["songs"]:
        validate_song(song)
        if song["song_id"].casefold() in seen:
            raise ValidationError("Duplicate song_id (case-insensitive filesystem identity)")
        seen.add(song["song_id"].casefold())
        for name in ("song_group_id", "artist_group_id", "near_duplicate_group_id", "audio_sha256"):
            value = song["audio"]["sha256"] if name == "audio_sha256" else song[name]
            if value is None:
                continue
            key = (name, value)
            if key in split_members and split_members[key] != song["split"]:
                raise ValidationError("Cross-split leakage for " + name + ": " + value)
            split_members[key] = song["split"]
    return dataset


def validate_plan(plan, song=None):
    keys(plan, ("schema", "song_id", "sample_rate", "num_samples", "motifs", "events", "provenance"), where="plan")
    if plan["schema"] != "choreobench.plan.v1":
        raise ValidationError("Unsupported plan schema")
    filename_identifier(plan["song_id"], "plan.song_id")
    integer(plan["sample_rate"], "sample_rate", 1)
    integer(plan["num_samples"], "num_samples", 1)
    if song is not None:
        validate_song(song)
        if (plan["song_id"], plan["sample_rate"], plan["num_samples"]) != (
                song["song_id"], song["audio"]["sample_rate"], song["audio"]["num_samples"]):
            raise ValidationError("Plan song identity or clock does not match dataset")
    if not isinstance(plan["provenance"], dict):
        raise ValidationError("provenance must be an object")
    try:
        finite_json(plan["provenance"])
    except ValueError as error:
        raise ValidationError(str(error)) from error
    if not isinstance(plan["motifs"], list):
        raise ValidationError("motifs must be a list")
    motifs = set()
    for motif in plan["motifs"]:
        keys(motif, ("motif_id", "description"), where="motif")
        identifier(motif["motif_id"], "motif_id")
        text(motif["description"], "motif.description")
        if motif["motif_id"] in motifs:
            raise ValidationError("Duplicate motif_id")
        motifs.add(motif["motif_id"])
    if not isinstance(plan["events"], list) or not plan["events"]:
        raise ValidationError("events must be nonempty")
    ids = set()
    occupied = {part: [] for part in TRACKS["body"]}
    for event in plan["events"]:
        keys(event, ("event_id", "start", "end", "track", "action", "attributes", "entry", "exit", "motif_id", "evidence"), where="event")
        identifier(event["event_id"], "event_id")
        if event["event_id"] in ids:
            raise ValidationError("Duplicate event_id")
        ids.add(event["event_id"])
        _interval(event, plan["num_samples"], "event")
        if not isinstance(event["track"], str) or event["track"] not in TRACKS:
            raise ValidationError("Invalid body track")
        text(event["action"], "event.action")
        for name in ("attributes", "entry", "exit"):
            _scalar_object(event[name], name)
        if event["motif_id"] is not None:
            identifier(event["motif_id"], "event.motif_id")
            if event["motif_id"] not in motifs:
                raise ValidationError("Unknown motif reference")
        if not isinstance(event["evidence"], list) or any(not isinstance(x, str) or not x.strip() for x in event["evidence"]):
            raise ValidationError("evidence must be a list of nonempty source references")
        for part in TRACKS[event["track"]]:
            occupied[part].append((event["start"], event["end"], event["event_id"]))
    for intervals in occupied.values():
        intervals.sort()
        last_end = -1
        for start, end, event_id in intervals:
            if start < last_end:
                raise ValidationError("Overlapping incompatible body scopes at " + event_id)
            last_end = end
    order = [(x["start"], x["track"], x["event_id"]) for x in plan["events"]]
    if order != sorted(order):
        raise ValidationError("Events must be ordered by (start, track, event_id)")
    return plan
