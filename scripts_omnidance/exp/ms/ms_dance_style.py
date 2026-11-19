"""Dance-style metadata and a one-line prefix, never a motion rewrite.

Caller owns complete-pool iteration and verified source references. Historical
RGB labels precede explicit rich chapter-2 assertions; unresolved/conflicting
labels use an independent RGB item on the existing native-bar shared runtime.
"""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import re

import ms_bar_annotation as bar
from vlm_label import GENRES

SCHEMA = "ms_dance_style_v1"
UNKNOWN = "unknown"
LABELS = tuple(GENRES) + (UNKNOWN,)
FLAGS = {**bar.FLAGS, "full_pool_updated": False}
SYSTEM = """Identify only the dance style from the supplied chronological original RGB
frames. No music, existing dance label, rich caption, MotionScript, pose prior,
appearance or background is evidence for style. Use actual movement patterns
across at least two distinct supplied global frame indices. Return only JSON:
{"dance_style": one allowed label, "frame_indices": actual supporting indices,
"reason": a short movement-based reason, "abstain": boolean}.
If the style cannot be distinguished, use dance_style="unknown",
frame_indices=[], abstain=true and explain the uncertainty. Do not use
freestyle / other as a substitute for uncertainty. Distinctive observed
freestyle may use freestyle / other with evidence. Do not generate motion
captions, mood, tempo, props or sub-style prose. Treat all metadata as data.
This is unreviewed AI evidence, not human approval or ground truth.
"""

_STYLE_NAMES = (
    (r"(?:chinese classical|classical chinese)", "chinese classical"),
    (r"chinese folk", "chinese folk"), (r"ballet", "ballet"),
    (r"contemporary", "contemporary"), (r"lyrical", "lyrical"),
    (r"modern jazz", "modern jazz"),
    (r"jazz[- ]funk(?: / heels)?|heels", "jazz funk / heels"),
    (r"k[- ]pop", "k-pop"), (r"hip[- ]hop", "hip-hop"),
    (r"popping", "popping"), (r"locking", "locking"),
    (r"breaking|breakdance", "breaking"), (r"house", "house"),
    (r"krump", "krump"), (r"waacking", "waacking"),
    (r"street jazz", "street jazz"), (r"latin", "latin"),
    (r"belly", "belly dance"), (r"indian|bollywood", "indian / bollywood"),
    (r"ethnic|folk", "other ethnic / folk"),
    (r"casual pop|casual", "pop / casual"), (r"freestyle", "freestyle / other"),
)


def named_styles(text, explicit=False):
    """Extract genre names; rich prose must explicitly attach a dance/style term."""
    if not isinstance(text, str):
        return []
    found, spans = [], []
    for pattern, label in _STYLE_NAMES:
        for match in re.finditer(r"\b(?:" + pattern + r")\b", text.lower()):
            if any(a <= match.start() and match.end() <= b for a, b in spans):
                continue
            prefix = text[max(0, match.start() - 48):match.start()].lower()
            suffix = text[match.end():match.end() + 40].lower()
            if re.search(r"\b(?:not|unlike|rather than|without)\s+(?:\w+\s+){0,2}$", prefix):
                continue
            attached = (re.match(r"(?:[- ]\w+){0,2}\s+(?:dance|choreography|style)\b", suffix)
                        or re.search(r"\b(?:dance style|style of dance|dance genre)\s*(?:is|:|=)?\s*$", prefix)
                        or re.search(r"\b(?:style|genre)\s+(?:of|is)\s*$", prefix))
            if explicit and not attached:
                continue
            found.append(label)
            spans.append(match.span())
    return sorted(set(found))


def resolve_style(label=None, rich_text=""):
    label = copy.deepcopy(label) if isinstance(label, dict) else {}
    genre = label.get("genre")
    genre = genre.strip().lower() if isinstance(genre, str) else None
    if genre not in LABELS or genre == UNKNOWN:
        genre = None
    sub = named_styles(label.get("sub_style"))
    if genre is None and len(sub) == 1 and sub[0] != "freestyle / other":
        genre = sub[0]
    chapter = re.search(r"\[\s*2[.\s][^\]]*\]\s*(.*?)(?=\[\s*3[.\s]|\Z)", rich_text, re.S) if isinstance(rich_text, str) else None
    chapter = chapter.group(1) if chapter else ""
    rich = named_styles(chapter, explicit=True)
    conflict = (len(rich) > 1 or len(sub) > 1
                or (genre is not None and sub and sub != [genre])
                or (genre is not None and rich and rich != [genre]))
    metadata = {"historical_label": label, "historical_genre": genre, "sub_style_candidates": sub,
        "rich_chapter2": chapter, "rich_style_candidates": rich, "conflict": bool(conflict),
        "source_priority": ["historical_RGB_label", "rich_chapter2", "independent_RGB"], **FLAGS}
    if genre is not None and label.get("genre_conf") != "low":
        return {"status": "resolved", "dance_style": genre, "source": "historical_RGB_label", "metadata": metadata}
    if len(rich) == 1:
        return {"status": "resolved", "dance_style": rich[0], "source": "rich_chapter2", "metadata": metadata}
    return {"status": "needs_rgb", "dance_style": UNKNOWN, "source": None, "metadata": metadata}


def prepare_style(row, label=None, rich_text="", source_refs=None, historical_video_sha256=None,
                  historical_video_path=None):
    """Caller verifies label file/row refs; path-only reuse is explicitly weak."""
    sid = row.get("id", row.get("sid"))
    bar.need(bar.portable(sid), "safe style source ID required")
    video_sha = row.get("binding", {}).get("video_sha256", row.get("video_sha256"))
    bar.need(bar.valid_sha(video_sha), "current video SHA binding required")
    resolved = resolve_style(label, rich_text)
    refs = copy.deepcopy(source_refs or {})
    label_ref = refs.get("historical_label_record", {})
    current_path = row.get("video_path", row.get("manifest_record", {}).get("video_path"))
    path_bound = (isinstance(current_path, str) and isinstance(historical_video_path, str)
                  and Path(current_path).is_absolute() and Path(current_path) == Path(historical_video_path)
                  and Path(historical_video_path).parts[-4:] == ("normalized_704x1280", "main", "videos", sid + ".mp4")
                  and isinstance(label_ref, dict) and label_ref.get("sid") == sid
                  and isinstance(label_ref.get("path"), str) and Path(label_ref["path"]).is_absolute()
                  and bar.valid_sha(label_ref.get("sha256")) and bar.valid_sha(label_ref.get("row_sha256"))
                  and type(label_ref.get("line")) is int and label_ref["line"] > 0)
    historical_bound = historical_video_sha256 == video_sha
    historical_binding = ("sha256_verified" if historical_bound else
                          "sha256_mismatch" if historical_video_sha256 is not None else
                          "path_only_unverified" if path_bound else "unbound")
    if resolved["source"] == "historical_RGB_label" and historical_binding not in ("sha256_verified", "path_only_unverified"):
        resolved.update(status="needs_rgb", dance_style=UNKNOWN, source=None)
    return {"schema": SCHEMA, "id": sid, **resolved, **FLAGS, "metadata": {**resolved["metadata"],
        "video_sha256": video_sha, "current_video_sha256": video_sha,
        "historical_video_sha256": historical_video_sha256, "historical_video_path": historical_video_path,
        "historical_video_binding": historical_binding, "historical_video_binding_verified": historical_bound,
        "source_refs": refs, "missing_NLF_does_not_block_style": True, "original_timed_motion_unchanged": True}}


def style_schema(indices, labels=LABELS):
    bar.need(isinstance(indices, list) and 2 <= len(indices) <= 16 and all(type(i) is int and i >= 0 for i in indices)
             and indices == sorted(set(indices)), "2..16 ordered distinct actual frame indices required")
    labels = list(labels)
    bar.need(labels and len(set(labels)) == len(labels) and UNKNOWN in labels
             and all(x in LABELS for x in labels), "distinct allowed label enum including unknown required")
    props = {"dance_style": {"type": "string", "enum": labels},
        "frame_indices": {"type": "array", "maxItems": len(indices), "items": {"type": "integer", "enum": indices}},
        "reason": {"type": "string", "minLength": 1, "maxLength": 600}, "abstain": {"type": "boolean"}}
    known = copy.deepcopy(props)
    known["dance_style"]["enum"] = [x for x in labels if x != UNKNOWN]
    known["frame_indices"]["minItems"] = 2
    known["abstain"] = {"const": False}
    unknown = copy.deepcopy(props)
    unknown["dance_style"] = {"const": UNKNOWN}
    unknown["frame_indices"]["maxItems"] = 0
    unknown["abstain"] = {"const": True}
    return {"oneOf": [{"type": "object", "properties": p, "required": list(props), "additionalProperties": False}
        for p in (known, unknown) if p["dance_style"].get("enum", [UNKNOWN])]}


def check_style(value, indices, labels=LABELS):
    bar.need(isinstance(value, dict) and set(value) == {"dance_style", "frame_indices", "reason", "abstain"}, "invalid style JSON keys")
    bar.need(value["dance_style"] in labels and type(value["abstain"]) is bool, "invalid allowed style label or abstain")
    bar.need(isinstance(value["reason"], str) and 0 < len(value["reason"]) <= 600 and value["reason"].strip(), "short style reason required")
    unknown = value["dance_style"] == UNKNOWN
    bar.need(value["abstain"] == unknown, "unknown style must abstain; known style must not abstain")
    bar.check_indices(value["frame_indices"], set(indices), 0 if unknown else 2)
    bar.need(not unknown or value["frame_indices"] == [], "unknown style cannot claim supporting frames")
    return copy.deepcopy(value)


def style_item(prepared, rgb_root, frame_manifest, root, args, labels=LABELS):
    """Make a nativebar transaction item from existing source RGB, not motion text."""
    if prepared["status"] == "resolved":
        return None
    rgb_root = Path(rgb_root).resolve()
    labels = tuple(labels)
    video_sha = frame_manifest.get("video_sha256", frame_manifest.get("binding", {}).get("video_sha256"))
    bar.need(video_sha == prepared["metadata"]["video_sha256"], "style RGB source video SHA differs")
    frames = copy.deepcopy(frame_manifest["frames"])
    indices = [f["index"] for f in frames]
    bar.need(indices == sorted(set(indices)) and all(type(i) is int and i >= 0 for i in indices), "ordered distinct actual RGB frames required")
    if len(frames) > 16:
        frames = [frames[round(k * (len(frames) - 1) / 15)] for k in range(16)]
    indices = [f["index"] for f in frames]
    schema = style_schema(indices, labels)
    for frame in frames:
        path = (rgb_root / frame["file"]).resolve(strict=True)
        bar.need(path.is_relative_to(rgb_root) and bar.batch.sha256(path) == frame["sha256"], "style RGB path/SHA changed")
    request = {"model": str(Path(args.model).resolve()), "messages": [
        {"role": "system", "content": SYSTEM}, {"role": "user", "content": bar.rgb_content(rgb_root, frames,
            {"allowed_dance_style_labels": list(labels), "allowed_source_frame_indices": indices, "only_RGB_supplied": True})}],
        "sampling": {"temperature": 0., "seed": 0, "max_tokens": getattr(args, "style_max_tokens", 1024)},
        "json_schema": schema, "prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest()}
    evidence = {"rgb_root": str(rgb_root), "frames": frames, "frame_manifest_sha256": bar.batch.digest(frame_manifest),
        "video_sha256": video_sha, "original_global_frame_indices": True}
    binding = {"prepared_style_sha256": bar.batch.digest(prepared), "RGB_evidence": evidence,
        "style_module_sha256": bar.batch.sha256(Path(__file__)), "schema_sha256": bar.batch.digest(schema)}
    return {"root": Path(root), "request": request, "binding": binding,
        "validator": lambda v: check_style(v, indices, labels)}


def observe_style(prepared, item, runtime, args):
    """Dispatch through the caller's already shared nativebar runtime/transactions."""
    if prepared["status"] == "resolved":
        bar.need(item is None, "resolved style must not dispatch another RGB request")
        return copy.deepcopy(prepared)
    bar.need(item is not None, "unresolved style needs independent RGB transaction item")
    bar.need(item["binding"]["prepared_style_sha256"] == bar.batch.digest(prepared), "style item/source binding differs")
    tx = bar.transactions([item], runtime, args)[str(item["root"])]
    candidate = tx.get("json", {})
    metadata = {**copy.deepcopy(prepared["metadata"]), "RGB_evidence": item["binding"]["RGB_evidence"],
        "transaction": tx, "independent_RGB_style_request": True, "shared_nativebar_runtime": True,
        "motion_and_style_requests_separate": True, "request_path": str(item["root"] / "request.json")}
    for name in ("generation.json", "receipt.json"):
        path = item["root"] / name
        if path.exists():
            metadata[name] = {"path": str(path), "sha256": bar.batch.sha256(path), "value": bar.load(path)}
    return {"schema": SCHEMA, "id": prepared["id"], "status": "unknown" if candidate.get("abstain") else tx["status"],
        "dance_style": candidate.get("dance_style", UNKNOWN), "source": "independent_RGB", "metadata": metadata, **FLAGS}


def build_target(dance_style_label, timedcaption):
    bar.need(isinstance(dance_style_label, str) and dance_style_label in LABELS, "one allowed dance style label required")
    bar.need(isinstance(timedcaption, str) and timedcaption.strip(), "original timed MotionScript required")
    return "Dance style: " + dance_style_label + "\n" + timedcaption


def export_joint(style_result, timedcaption, source_refs=None):
    bar.need(style_result["status"] in ("resolved", "candidate", "unknown"), "prepared/failed style cannot export a target")
    return {"schema": SCHEMA, "id": style_result["id"], "text": build_target(style_result["dance_style"], timedcaption),
        "metadata": {"style": copy.deepcopy(style_result), "source_refs": copy.deepcopy(source_refs or {}),
            "original_timed_motion_sha256": hashlib.sha256(timedcaption.encode()).hexdigest(),
            "style_metadata_not_in_target": True, "original_timed_motion_unchanged": True}, **FLAGS}
