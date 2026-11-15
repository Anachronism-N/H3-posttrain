"""Shared versioned review contracts; no model output is ground truth."""
import copy
import hashlib
import json
import math
from pathlib import Path
import urllib.request
import wave

CRITERIA = ("musical_fit", "motif_development", "action_executability", "boundary_connection")


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result

    def bad(value):
        raise ValueError("Non-finite JSON number: " + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=bad)


def read_json(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def finite(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(label + " must be finite")
    return value


def compact(events, limit=0, hide_motif=True, scope=None):
    """Zero means full text. Times stay in global seconds even for audio crops."""
    if type(limit) is not int or limit < 0:
        raise ValueError("action-chars must be a nonnegative integer; zero means full text")
    result, ids = [], set()
    for e in events:
        event_id = e.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in ids:
            raise ValueError("Event IDs must be nonempty and unique")
        ids.add(event_id)
        start, end = finite(e.get("start"), "start"), finite(e.get("end"), "end")
        if start < 0 or end <= start:
            raise ValueError("Invalid event interval in seconds")
        if scope and not (start < scope["audio_global_end_seconds"] and
                          end > scope["audio_global_start_seconds"]):
            continue
        action = e.get("action")
        if not isinstance(action, str) or not action.strip():
            raise ValueError("Every event needs action text")
        row = dict(event_id=event_id, start=start, end=end, track=e.get("track"),
                   action=action if limit == 0 else action[:limit])
        if not hide_motif:
            row["motif"] = e.get("motif_id")
        if scope:
            row["observed_interval_seconds"] = [max(start, scope["audio_global_start_seconds"]),
                                                min(end, scope["audio_global_end_seconds"])]
        result.append(row)
    return result


def check_pair(pair):
    if pair.get("time_unit") != "seconds":
        raise ValueError("Review pair must declare time_unit=seconds; rebuild legacy bundles")
    for key in ("pair_id", "song_id", "comparison"):
        if not isinstance(pair.get(key), str) or not pair[key]:
            raise ValueError("Missing review identifier: " + key)
    for key in ("A", "B"):
        if not compact(pair[key]):
            raise ValueError("Cannot review an empty plan")


def _reason(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Review must include a nonempty evidence reason")


def validate_scores(value, criteria=CRITERIA, with_reason=True):
    expected = set(criteria) | ({"reason"} if with_reason else set())
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Review score fields differ from the frozen criteria")
    for key in criteria:
        if type(value[key]) is not int or not 1 <= value[key] <= 5:
            raise ValueError(key + " must be an integer in [1,5], not boolean")
    if with_reason:
        _reason(value["reason"])
    return value


def validate_pairwise(value, audio=False):
    expected = {"A", "B", "preferred"} | ({"heard"} if audio else set())
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Pairwise response has missing or extra fields")
    validate_scores(value["A"])
    validate_scores(value["B"])
    if value["preferred"] not in ("A", "B", "tie"):
        raise ValueError("preferred must be A/B/tie")
    if audio:
        validate_heard(value["heard"])
    return value


def validate_heard(value):
    if not isinstance(value, list) or len(value) < 2 or any(
            not isinstance(x, str) or not x.strip() for x in value):
        raise ValueError("heard must contain at least two nonempty observations")


def validate_absolute(value):
    if not isinstance(value, dict) or set(value) != {"heard", "scores", "reason"}:
        raise ValueError("Absolute response has missing or extra fields")
    validate_scores(value["scores"], with_reason=False)
    validate_heard(value["heard"])
    _reason(value["reason"])
    return value


def response_text(response):
    if not isinstance(response, dict) or not isinstance(response.get("choices"), list) or len(response["choices"]) != 1:
        raise ValueError("Expected one chat-completion choice")
    choice = response["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("Incomplete judge response: " + str(choice.get("finish_reason")))
    value = choice.get("message", {}).get("content")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing judge response text")
    return value


def swapped(value):
    result = copy.deepcopy(value)
    result["A"], result["B"] = result["B"], result["A"]
    result["preferred"] = {"A": "B", "B": "A", "tie": "tie"}[value["preferred"]]
    return result


def consistency(first, second):
    result = {}
    for c in CRITERIA:
        a, b = first["A"][c] - first["B"][c], second["A"][c] - second["B"][c]
        result[c] = dict(original=a, swapped=b,
                         consistent=(a == b == 0) or (a > 0 and b > 0) or (a < 0 and b < 0))
    return result


def audio_scope(path, offset=0.0, expected_end=None):
    """Metadata describes bytes sent, not proof the service encoded all of them."""
    offset = finite(offset, "audio_global_start_seconds")
    if offset < 0:
        raise ValueError("Audio global offset must be nonnegative")
    with wave.open(str(path), "rb") as wav:
        rate, count, channels = wav.getframerate(), wav.getnframes(), wav.getnchannels()
    if count <= 0 or rate <= 0:
        raise ValueError("Audio is empty")
    end = offset + count / rate
    if expected_end is not None and abs(finite(expected_end, "audio end") - end) > 1 / rate:
        raise ValueError("Declared audio interval does not match WAV frames")
    return dict(time_unit="seconds", audio_global_start_seconds=offset, audio_global_end_seconds=end,
                audio_duration_seconds=count / rate, sample_rate=rate, num_samples=count, channels=channels,
                audio_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                server_encoding_verified=False,
                verification_note="Sent WAV scope only; heard text is not encoding or listening proof.")


def resolve_audio(song, root=None, manifest=None):
    if manifest is None:
        if root is None:
            raise ValueError("Provide audio-root or audio-manifest")
        path = Path(root) / (song + ".wav")
        return path, audio_scope(path)
    # Manifest is keyed by song_id; each entry binds a full song or explicit crop.
    if song not in manifest:
        raise ValueError("Audio manifest does not cover song " + song)
    row = manifest[song]
    path = Path(row["audio_path"])
    scope = audio_scope(path, row["audio_global_start_seconds"], row["audio_global_end_seconds"])
    if row.get("audio_sha256") != scope["audio_sha256"]:
        raise ValueError("Audio manifest SHA mismatch")
    return path, scope


class AuditLog:
    """Exclusive per-run log writes raw responses before parsing, including failures."""
    def __init__(self, out):
        self.out = Path(out)
        self.path = self.out.with_suffix(self.out.suffix + ".calls.jsonl")
        if self.out.exists():
            raise FileExistsError(str(self.out))
        self.out.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("x", encoding="utf-8"):
            pass

    def append(self, value):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

    def request(self, url, model, messages, *, max_tokens, seed, call_id, safe_payload):
        body = dict(model=model, messages=messages, temperature=0, max_tokens=max_tokens,
                    seed=seed, response_format={"type": "json_object"})
        self.append(dict(call_id=call_id, phase="started", model=model,
                         request_payload=safe_payload, seed=seed, max_output_tokens=max_tokens))
        request = urllib.request.Request(url.rstrip("/") + "/chat/completions",
                                         data=json.dumps(body, ensure_ascii=False).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=1800) as reply:
                raw = reply.read().decode("utf-8")
            # Save malformed JSON/API errors too, before any validation.
            self.append(dict(call_id=call_id, phase="returned", raw_response=raw))
            return strict_json(raw)
        except Exception as error:
            self.append(dict(call_id=call_id, phase="failed", error=repr(error)))
            raise
