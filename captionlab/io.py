"""Explicit UTF-8, finite JSON, and content identities for experiment artifacts."""
import hashlib
import json
from pathlib import Path


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def object_sha(value):
    return text_sha(json.dumps(value, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":"), allow_nan=False))


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def loads(text):
    def invalid(value):
        raise ValueError("Non-finite JSON: " + value)
    return json.loads(text, object_pairs_hook=_unique, parse_constant=invalid)


def read_json(path):
    return loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    # Exclusive creation prevents accidental replacement of a frozen experiment.
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(data)
