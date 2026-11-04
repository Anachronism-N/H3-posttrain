"""Metadata-blinded review packages and hash-bound external review imports.

Hashes bind artifacts, not the identity or honesty of a reviewer. audio_seen is a
reviewer's declaration, never programmatic proof that somebody listened.
"""

import copy
import json
import random
import re
from pathlib import PurePath

from .editing import public_edit_task, validate_edit_task
from .io import object_sha
from .schema import ValidationError, validate_plan, validate_song


CRITERIA = (
    {"criterion_id": "musical_fit", "requires_audio": True,
     "description": "How the planned choreography fits this audio; not observed motion synchronization."},
    {"criterion_id": "long_range_structure", "requires_audio": False,
     "description": "Planned contrast, development, motif recurrence and ending structure."},
    {"criterion_id": "planned_executability", "requires_audio": False,
     "description": "Expert judgment of the written plan; not measured biomechanics."},
    {"criterion_id": "edit_target_satisfaction", "requires_audio": False,
     "description": "Whether the edit meets the public instruction."},
    {"criterion_id": "edit_preservation", "requires_audio": False,
     "description": "Whether protected and unaffected action facts are retained."},
    {"criterion_id": "planned_boundary_quality", "requires_audio": False,
     "description": "Compatibility of planned entry and exit states; not rendered continuity."},
)
_CRITERION_IDS = {criterion["criterion_id"] for criterion in CRITERIA}
_AUDIO_CRITERIA = {criterion["criterion_id"] for criterion in CRITERIA if criterion["requires_audio"]}


def _require(condition, message):
    if not condition:
        raise ValidationError(message)


def _exact(value, fields, label):
    _require(isinstance(value, dict) and set(value) == set(fields), label + " has missing or unexpected fields")


def _nonempty(value, label):
    _require(isinstance(value, str) and bool(value.strip()), label + " must be nonempty text")


def _validate_source_binding(value):
    """Check organizer declarations; hashes do not authenticate their source."""
    _exact(value, {"kind", "sha256", "dataset_sha256", "synthetic"}, "source_binding")
    _require(isinstance(value["kind"], str) and value["kind"] in {"submission", "run"},
             "source_binding kind must be submission/run")
    for field in ("sha256", "dataset_sha256"):
        _require(isinstance(value[field], str) and re.fullmatch(r"[0-9a-f]{64}", value[field]) is not None,
                 "source_binding " + field + " must be lowercase SHA256")
    _require(type(value["synthetic"]) is bool, "source_binding synthetic must be boolean")
    return value


def _item_sha(item):
    return object_sha({key: value for key, value in item.items() if key != "item_sha256"})


def _blind_plan(plan, song_alias, event_ids, motif_ids):
    result = copy.deepcopy(plan)
    result["song_id"] = song_alias
    result["provenance"] = {}
    for motif in result["motifs"]:
        motif["motif_id"] = motif_ids[motif["motif_id"]]
    for event in result["events"]:
        event["event_id"] = event_ids[event["event_id"]]
        event["motif_id"] = motif_ids[event["motif_id"]] if event["motif_id"] is not None else None
        event["evidence"] = []
    result["events"].sort(key=lambda event: (event["start"], event["track"], event["event_id"]))
    validate_plan(result)
    return result


def _blind_song(song, alias):
    music = copy.deepcopy(song["music"])
    music["source"] = "public_music_predictions"
    repeat_ids = {}
    for index, section in enumerate(music["sections"], 1):
        section["section_id"] = "section_{:04d}".format(index)
        if section["repeat_group"] is not None:
            original = section["repeat_group"]
            repeat_ids.setdefault(original, "repeat_{:04d}".format(len(repeat_ids) + 1))
            section["repeat_group"] = repeat_ids[original]
    suffix = PurePath(song["audio"]["path"].replace("\\", "/")).suffix.lower()
    if suffix not in {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus"}:
        suffix = ".audio"
    audio = copy.deepcopy(song["audio"])
    audio["path"] = "assets/" + audio["sha256"] + suffix
    return {"song_id": alias, "brief": song["brief"], "audio": audio, "music": music}


def build_review_pack(items, seed=0):
    """Return separate public and private JSON objects; this function copies no audio.

    Each input item is {song, plan, method_id, source_plan?, edit_task?, source_binding?}. The caller
    must retain private bindings separately and provide the hash-bound audio at the
    public asset path or via an asset index. Private goals never enter this pack.
    """
    _require(isinstance(items, list) and bool(items), "review items must be a nonempty list")
    _require(type(seed) is int, "review seed must be an integer")
    checked, songs, methods = [], {}, set()
    for item in items:
        _require(isinstance(item, dict), "review item must be an object")
        _require({"song", "plan", "method_id"}.issubset(item) and
                 set(item).issubset({"song", "plan", "method_id", "source_plan", "edit_task", "source_binding"}),
                 "review item has missing or unexpected fields")
        validate_song(item["song"])
        validate_plan(item["plan"], item["song"])
        _nonempty(item["method_id"], "method_id")
        if "source_binding" in item:
            _validate_source_binding(item["source_binding"])
        methods.add(item["method_id"])
        original_song_id = item["song"]["song_id"]
        if original_song_id in songs:
            _require(object_sha(songs[original_song_id]) == object_sha(item["song"]),
                     "same song_id has inconsistent song/audio evidence")
        songs[original_song_id] = item["song"]
        has_source, has_task = item.get("source_plan") is not None, item.get("edit_task") is not None
        _require(has_source == has_task, "edit review requires both source_plan and edit_task")
        if has_source:
            validate_plan(item["source_plan"], item["song"])
            validate_edit_task(item["edit_task"], item["source_plan"])
        checked.append(copy.deepcopy(item))
    aliases = {song_id: "song_{:04d}".format(index) for index, song_id in enumerate(sorted(songs), 1)}
    random.Random(seed).shuffle(checked)
    public = {
        "schema": "choreobench.review_pack.v1",
        "criteria": copy.deepcopy(list(CRITERIA)), "rating_scale": {"minimum": 1, "maximum": 5},
        "items": [],
        "limitations": [
            "Metadata blinded; free natural-language content still requires independent blinding review.",
            "Audio paths are an asset contract, not proof that audio has been copied or listened to.",
            "Written plans cannot establish physical quality or observed beat synchronization.",
        ],
    }
    private = {"schema": "choreobench.review_bindings.v1", "seed": seed,
               "public_pack_sha256": None, "items": []}
    for index, original in enumerate(checked, 1):
        song, plan = original["song"], original["plan"]
        source, task = original.get("source_plan"), original.get("edit_task")
        all_plans = [plan] + ([source] if source is not None else [])
        event_names = sorted({event["event_id"] for candidate in all_plans for event in candidate["events"]})
        motif_names = sorted({motif["motif_id"] for candidate in all_plans for motif in candidate["motifs"]})
        event_ids = {name: "event_{:04d}".format(i) for i, name in enumerate(event_names, 1)}
        motif_ids = {name: "motif_{:04d}".format(i) for i, name in enumerate(motif_names, 1)}
        alias = aliases[song["song_id"]]
        review_id = "review_{:06d}".format(index)
        public_song = _blind_song(song, alias)
        public_source = _blind_plan(source, alias, event_ids, motif_ids) if source is not None else None
        public_task = None
        if task is not None:
            public_task = public_edit_task(task)
            public_task.update({"song_id": alias, "task_id": "task_" + review_id,
                                "source_plan_sha256": object_sha(public_source)})
            for name in ("allowed_event_ids", "repair_event_ids"):
                public_task[name] = [event_ids[event_id] for event_id in task[name]]
            for protection in public_task["protected"]:
                if protection["event_id"] != "*":
                    protection["event_id"] = event_ids[protection["event_id"]]
        public_item = {
            "review_id": review_id, "song": public_song,
            "plan": _blind_plan(plan, alias, event_ids, motif_ids),
            "source_plan": public_source, "edit_task": public_task,
        }
        public_item["item_sha256"] = _item_sha(public_item)
        public["items"].append(public_item)
        private["items"].append({
            "review_id": review_id, "item_sha256": public_item["item_sha256"],
            "song_id": song["song_id"], "method_id": original["method_id"],
            "song_sha256": object_sha(song), "plan_sha256": object_sha(plan),
            "source_plan_sha256": object_sha(source) if source is not None else None,
            "edit_task_sha256": object_sha(task) if task is not None else None,
            "audio_source_path": song["audio"]["path"],
            "public_audio_path": public_song["audio"]["path"], "audio_sha256": song["audio"]["sha256"],
            "event_id_map": event_ids, "motif_id_map": motif_ids,
            "original": copy.deepcopy(original),
        })
        if "source_binding" in original:
            private["items"][-1]["source_binding_sha256"] = object_sha(original["source_binding"])
    serialized = json.dumps(public, ensure_ascii=False, sort_keys=True)
    # Reject explicitly named methods instead of silently editing action content to hide them.
    for method in methods:
        pattern = r"(?<![A-Za-z0-9_])" + re.escape(method) + r"(?![A-Za-z0-9_])"
        _require(re.search(pattern, serialized) is None,
                 "method identifier survives in public content; independently blind the text first")
    private["public_pack_sha256"] = object_sha(public)
    return {"public": public, "private": private}


def validate_review_pack(pack):
    _exact(pack, {"schema", "criteria", "rating_scale", "items", "limitations"}, "review pack")
    _require(pack["schema"] == "choreobench.review_pack.v1", "unsupported review pack schema")
    _require(object_sha(pack["criteria"]) == object_sha(list(CRITERIA)) and
             object_sha(pack["rating_scale"]) == object_sha({"minimum": 1, "maximum": 5}),
             "review criteria or rating scale changed")
    _require(isinstance(pack["items"], list) and bool(pack["items"]), "review pack items must be nonempty")
    seen = set()
    for item in pack["items"]:
        _exact(item, {"review_id", "song", "plan", "source_plan", "edit_task", "item_sha256"}, "public review item")
        _nonempty(item["review_id"], "review_id")
        _require(item["review_id"] not in seen, "duplicate review_id")
        seen.add(item["review_id"])
        _require(item["item_sha256"] == _item_sha(item), "public review item hash mismatch")
        validate_plan(item["plan"])
        _require(item["plan"]["provenance"] == {} and all(not event["evidence"] for event in item["plan"]["events"]),
                 "review plan contains private provenance/evidence")
    return pack


def blank_review_template(pack, reviewer_id="FILL_REVIEWER_ID", origin="human", model_id=None):
    """A fill-in form. Null ratings mean pending, not a neutral or passing score."""
    validate_review_pack(pack)
    _nonempty(reviewer_id, "reviewer_id")
    _require(isinstance(origin, str) and origin in {"human", "model_judge"}, "origin must be human or model_judge")
    if origin == "human":
        _require(model_id is None, "human reviews cannot declare a model identity")
    else:
        _nonempty(model_id, "model_id")
    return {
        "schema": "choreobench.reviews.v1", "pack_sha256": object_sha(pack),
        "reviews": [
            {"review_id": item["review_id"], "item_sha256": item["item_sha256"],
             "reviewer_id": reviewer_id, "origin": origin, "model_id": model_id,
             "audio_seen": False, "ratings": {criterion: None for criterion in sorted(_CRITERION_IDS)},
             "comments": ""}
            for item in pack["items"]
        ],
    }


def _validate_bindings(pack, bindings):
    _exact(bindings, {"schema", "seed", "public_pack_sha256", "items"}, "private bindings")
    _require(bindings["schema"] == "choreobench.review_bindings.v1", "unsupported private bindings schema")
    _require(bindings["public_pack_sha256"] == object_sha(pack), "bindings do not match this review pack")
    _require(isinstance(bindings["items"], list), "private bindings items must be a list")
    public_items = {item["review_id"]: item for item in pack["items"]}
    seen = set()
    for item in bindings["items"]:
        _require(isinstance(item, dict) and item.get("review_id") in public_items,
                 "private binding references an unknown item")
        review_id = item["review_id"]
        _require(review_id not in seen, "duplicate private binding")
        seen.add(review_id)
        _require(item.get("item_sha256") == public_items[review_id]["item_sha256"], "private item hash mismatch")
        original = item.get("original")
        _require(isinstance(original, dict) and {"song", "plan", "method_id"}.issubset(original),
                 "private binding lacks original evidence")
        if "source_binding" in original:
            _validate_source_binding(original["source_binding"])
            _require(item.get("source_binding_sha256") == object_sha(original["source_binding"]),
                     "private source binding hash mismatch")
        else:
            _require("source_binding_sha256" not in item, "source binding hash has no bound original declaration")
        for name in ("song", "plan", "source_plan", "edit_task"):
            expected = object_sha(original[name]) if original.get(name) is not None else None
            _require(item.get(name + "_sha256") == expected, "original {} hash mismatch".format(name))
        _require(item.get("method_id") == original["method_id"] and item.get("song_id") == original["song"]["song_id"],
                 "private method/song binding mismatch")
        _require(item.get("audio_sha256") == original["song"]["audio"]["sha256"] == public_items[review_id]["song"]["audio"]["sha256"],
                 "audio binding mismatch")
        validate_song(original["song"])
        validate_plan(original["plan"], original["song"])
        all_plans = [original["plan"]]
        if original.get("source_plan") is not None:
            validate_plan(original["source_plan"], original["song"])
            validate_edit_task(original["edit_task"], original["source_plan"])
            all_plans.append(original["source_plan"])
        event_names = sorted({event["event_id"] for candidate in all_plans for event in candidate["events"]})
        motif_names = sorted({motif["motif_id"] for candidate in all_plans for motif in candidate["motifs"]})
        event_ids = {name: "event_{:04d}".format(i) for i, name in enumerate(event_names, 1)}
        motif_ids = {name: "motif_{:04d}".format(i) for i, name in enumerate(motif_names, 1)}
        _require(item.get("event_id_map") == event_ids and item.get("motif_id_map") == motif_ids,
                 "private ID mapping changed")
        public_item = public_items[review_id]
        alias = public_item["song"]["song_id"]
        _require(object_sha(public_item["song"]) == object_sha(_blind_song(original["song"], alias)),
                 "public song no longer matches original evidence")
        _require(object_sha(public_item["plan"]) == object_sha(_blind_plan(original["plan"], alias, event_ids, motif_ids)),
                 "public plan no longer matches the bound original plan")
        expected_source, expected_task = None, None
        if original.get("source_plan") is not None:
            expected_source = _blind_plan(original["source_plan"], alias, event_ids, motif_ids)
            expected_task = public_edit_task(original["edit_task"])
            expected_task.update({"song_id": alias, "task_id": "task_" + review_id,
                                  "source_plan_sha256": object_sha(expected_source)})
            for name in ("allowed_event_ids", "repair_event_ids"):
                expected_task[name] = [event_ids[event_id] for event_id in original["edit_task"][name]]
            for protection in expected_task["protected"]:
                if protection["event_id"] != "*":
                    protection["event_id"] = event_ids[protection["event_id"]]
        _require(object_sha(public_item["source_plan"]) == object_sha(expected_source) and
                 object_sha(public_item["edit_task"]) == object_sha(expected_task),
                 "public source/task no longer matches the bound originals")
        _require(item.get("audio_source_path") == original["song"]["audio"]["path"] and
                 item.get("public_audio_path") == public_item["song"]["audio"]["path"], "audio path binding mismatch")
    _require(seen == set(public_items), "private bindings must cover every public item")


def import_reviews(pack, bindings, reviews):
    """Validate imported declarations and keep absent/unfilled reviews pending.

    No automatic consensus, semantic pass, musical score, or human authentication is
    inferred. Model judges are counted separately from declared human reviewers.
    """
    validate_review_pack(pack)
    _validate_bindings(pack, bindings)
    _exact(reviews, {"schema", "pack_sha256", "reviews"}, "reviews document")
    _require(reviews["schema"] == "choreobench.reviews.v1", "unsupported reviews schema")
    _require(reviews["pack_sha256"] == object_sha(pack), "reviews bind a different pack")
    _require(isinstance(reviews["reviews"], list), "reviews must be a list")
    public_items = {item["review_id"]: item for item in pack["items"]}
    accepted = {review_id: [] for review_id in public_items}
    seen = set()
    for row in reviews["reviews"]:
        _exact(row, {"review_id", "item_sha256", "reviewer_id", "origin", "model_id", "audio_seen", "ratings", "comments"}, "review")
        _require(isinstance(row["review_id"], str) and row["review_id"] in public_items, "review references unknown item")
        _require(row["item_sha256"] == public_items[row["review_id"]]["item_sha256"], "review item hash mismatch")
        _nonempty(row["reviewer_id"], "reviewer_id")
        _require(row["reviewer_id"] != "FILL_REVIEWER_ID", "fill the real declared reviewer_id before importing")
        _require(isinstance(row["origin"], str) and row["origin"] in {"human", "model_judge"}, "unsupported review origin")
        if row["origin"] == "human":
            _require(row["model_id"] is None, "human review cannot declare model_id")
        else:
            _nonempty(row["model_id"], "model_id")
        _require(type(row["audio_seen"]) is bool, "audio_seen must be a boolean declaration")
        _require(isinstance(row["comments"], str), "review comments must be text")
        identity = (row["review_id"], row["reviewer_id"])
        _require(identity not in seen, "duplicate reviewer/item; do not count duplicates as independent judgments")
        seen.add(identity)
        _require(isinstance(row["ratings"], dict) and set(row["ratings"]).issubset(_CRITERION_IDS), "unknown review criterion")
        for criterion, value in row["ratings"].items():
            _require(value is None or (type(value) is int and 1 <= value <= 5), "ratings must be integer 1..5 or null")
            if value is not None and criterion in _AUDIO_CRITERIA:
                _require(row["audio_seen"], "musical_fit requires an explicit audio_seen declaration")
            if value is not None and criterion.startswith("edit_"):
                _require(public_items[row["review_id"]]["edit_task"] is not None,
                         "edit ratings require an edit task/source")
        accepted[row["review_id"]].append(copy.deepcopy(row))
    private_items = {item["review_id"]: item for item in bindings["items"]}
    results = []
    for review_id in sorted(public_items):
        rows = accepted[review_id]
        filled = [row for row in rows if any(value is not None for value in row["ratings"].values())]
        applicable = _CRITERION_IDS if public_items[review_id]["edit_task"] is not None else {
            criterion for criterion in _CRITERION_IDS if not criterion.startswith("edit_")}
        criteria_status = {
            criterion: ("not_applicable" if criterion not in applicable else
                        "reviewed" if any(row["ratings"].get(criterion) is not None for row in rows) else "pending")
            for criterion in sorted(_CRITERION_IDS)
        }
        def review_status(subset):
            present = {criterion for row in subset for criterion, value in row["ratings"].items() if value is not None}
            return "reviewed" if applicable.issubset(present) else "partial" if present else "pending"
        results.append({
            "review_id": review_id, "song_id": private_items[review_id]["song_id"],
            "method_id": private_items[review_id]["method_id"],
            "plan_sha256": private_items[review_id]["plan_sha256"],
            "source_binding": copy.deepcopy(private_items[review_id]["original"].get("source_binding")),
            "scientific_result_status": (
                "unknown" if "source_binding" not in private_items[review_id]["original"] else
                "synthetic" if private_items[review_id]["original"]["source_binding"]["synthetic"] else
                "declared_real_not_audited"
            ),
            "status": review_status(rows),
            "human_status": review_status([row for row in rows if row["origin"] == "human"]),
            "model_judge_status": review_status([row for row in rows if row["origin"] == "model_judge"]),
            "criteria_status": criteria_status,
            "human_review_count": sum(row["origin"] == "human" for row in filled),
            "model_judge_review_count": sum(row["origin"] == "model_judge" for row in filled),
            "reviews": rows,
        })
    return {
        "schema": "choreobench.review_report.v1", "pack_sha256": object_sha(pack),
        "bindings_sha256": object_sha(bindings), "reviews_sha256": object_sha(reviews),
        "items": results, "pending_item_count": sum(item["status"] != "reviewed" for item in results),
        "note": "Origins, audio_seen and source bindings are declarations, not authenticated identities, source audits or proof of listening. Ratings are external judgments, not verified semantic truth or measured motion synchronization.",
    }
