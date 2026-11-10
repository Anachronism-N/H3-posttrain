"""Stream caption JSONL evidence; language proxies never approve supervision.

No models, relabeling, training selection or source-file changes occur here.
Memory scales with one record plus the necessary ID/source-group indexes.
"""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import re
import unicodedata

from .io import loads, write_json


_HEADER = re.compile(r"^\s*Dance\s+style\s*:", re.I)
_TIMESTAMP = re.compile(r"^\s*\[\s*\d+(?:\.\d+)?\s*s?\s*-\s*\d+(?:\.\d+)?\s*s?\s*\]\s*", re.I)
_SENTENCES = re.compile(r"(?<=[.;])\s+|\r?\n+")
_WORDS = re.compile(r"[A-Za-z]+(?:['\u2019-][A-Za-z]+)*")
_ACTION_WORDS = frozenset("""move moves moved moving step steps stepped stepping turn turns turned turning
    pivot pivots pivoted pivoting travel travels traveled travelling walk walks walked walking
    lunge lunges lunged lunging shift shifts shifted shifting face faces faced facing
    reach reaches reached reaching extend extends extended extending lift lifts lifted lifting
    raise raises raised raising swing swings swung swinging rotate rotates rotated rotating
    lean leans leaned leaning crouch crouches crouched crouching jump jumps jumped jumping
    slide slides slid sliding cross crosses crossed crossing open opens opened opening
    close closes closed closing bend bends bent bending straighten straightens straightened straightening""".split())


def _body(caption):
    return "\n".join(_TIMESTAMP.sub("", line) for line in caption.splitlines() if not _HEADER.match(line)).strip()


def _sentences(body):
    return [part.strip() for part in _SENTENCES.split(body) if part.strip()]


def _script_status(body):
    if not body.strip():
        return "empty_body"
    latin = other = 0
    for character in body:
        if character.isalpha():
            if "LATIN" in unicodedata.name(character, ""):
                latin += 1
            else:
                other += 1
    if not latin and not other:
        return "no_alphabetic_characters"
    if not other:
        return "latin_letters_only"
    return "mixed_latin_and_non_latin_letters" if latin else "non_latin_letters_only"


def _proxy():
    return collections.Counter(texts=0, nonempty_sentence_texts=0, nonempty_word_texts=0,
        sentence_count=0, duplicate_sentence_excess=0, word_count=0, eight_gram_count=0,
        duplicate_eight_gram_excess=0, action_word_hits=0, sentence_rate_sum=0.)


def _add_proxy(total, body):
    sentences = _sentences(body)
    words = [word.lower() for word in _WORDS.findall(body)]
    excess = len(sentences) - len(set(sentences))
    grams = collections.Counter(tuple(words[index:index + 8]) for index in range(max(0, len(words) - 7)))
    total.update(texts=1, nonempty_sentence_texts=int(bool(sentences)), nonempty_word_texts=int(bool(words)),
        sentence_count=len(sentences), duplicate_sentence_excess=excess, word_count=len(words),
        eight_gram_count=max(0, len(words) - 7), duplicate_eight_gram_excess=sum(count - 1 for count in grams.values()),
        action_word_hits=sum(word in _ACTION_WORDS for word in words))
    if sentences:
        total["sentence_rate_sum"] += excess / len(sentences)
    return set(sentences)


def _finish_proxy(total):
    result = dict(total)
    rate_sum = result.pop("sentence_rate_sum")
    result["pooled_sentence_duplicate_rate"] = total["duplicate_sentence_excess"] / total["sentence_count"] if total["sentence_count"] else None
    result["mean_per_text_sentence_duplicate_rate"] = rate_sum / total["nonempty_sentence_texts"] if total["nonempty_sentence_texts"] else None
    result["pooled_rep8"] = total["duplicate_eight_gram_excess"] / total["eight_gram_count"] if total["eight_gram_count"] else None
    result["action_word_hits_per_1000_words"] = 1000 * total["action_word_hits"] / total["word_count"] if total["word_count"] else None
    return result


def _finite(value):
    if isinstance(value, str):
        value.encode("utf-8")  # Reject escaped lone surrogates as malformed text.
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite JSON number")
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(key)
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


def _identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _field(record, name, counts):
    value = record.get(name)
    if value is None:
        counts["missing_" + name] += 1
    elif not isinstance(value, str) or not value.strip():
        counts["invalid_" + name] += 1
    else:
        return value
    return None


def audit_labels(input_paths):
    """Return aggregate observations, never a list of selected training records.

    Every physical line enters line_count, including blank/invalid JSON lines.
    Metadata is counted only for decoded objects; each field has its own missing
    count. Split names are exact opaque strings, with no alias inference.
    """
    if not isinstance(input_paths, (list, tuple)) or not input_paths:
        raise ValueError("At least one input path is required")
    paths = [Path(path).resolve(strict=True) for path in input_paths]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate input path would double-count the same JSONL file")
    counts = collections.Counter(line_count=0, json_object_records=0, invalid_utf8_lines=0,
        invalid_json_lines=0, non_object_json_lines=0, missing_id=0, invalid_id=0,
        missing_split=0, invalid_split=0, missing_source_group_id=0, invalid_source_group_id=0,
        duplicate_id_occurrences=0, invalid_motion_caption_records=0, whole_caption_available_records=0,
        missing_caption_version=0, invalid_caption_version=0,
        structured_windows_available_records=0, structured_windows_unavailable_records=0,
        malformed_structured_windows_records=0, cross_window_duplicate_sentence_excess=0,
        sum_window_unique_sentences=0)
    ids, duplicate_ids, groups = {}, set(), {}
    split_counts, versions, scripts = collections.Counter(), collections.Counter(), collections.Counter()
    window_scripts = collections.Counter()
    whole, windows = _proxy(), _proxy()
    inputs, identities, error_examples = [], [], []

    def example(index, line, kind):
        if len(error_examples) < 20:
            error_examples.append(dict(input_index=index, line_number=line, kind=kind))

    for index, path in enumerate(paths):
        digest, size, lines = hashlib.sha256(), 0, 0
        with path.open("rb") as stream:
            before = _identity(os.fstat(stream.fileno()))
            for lines, raw in enumerate(stream, 1):
                digest.update(raw); size += len(raw); counts["line_count"] += 1
                try:
                    decoded = raw.decode("utf-8")
                except UnicodeDecodeError:
                    counts["invalid_utf8_lines"] += 1; example(index, lines, "invalid_utf8")
                    continue
                try:
                    record = loads(decoded)
                    _finite(record)
                except (ValueError, RecursionError):
                    counts["invalid_json_lines"] += 1; example(index, lines, "invalid_json")
                    continue
                if not isinstance(record, dict):
                    counts["non_object_json_lines"] += 1; example(index, lines, "non_object_json")
                    continue
                counts["json_object_records"] += 1
                identifier = _field(record, "id", counts)
                split = _field(record, "split", counts)
                group = _field(record, "source_group_id", counts)
                if split is not None:
                    split_counts[split] += 1
                if identifier is not None:
                    if identifier in ids:
                        duplicate_ids.add(identifier); counts["duplicate_id_occurrences"] += 1
                    ids.setdefault(identifier, set())
                    if split is not None:
                        ids[identifier].add(split)
                if group is not None:
                    groups.setdefault(group, set())
                    if split is not None:
                        groups[group].add(split)
                version = record.get("caption_version")
                if version is None:
                    counts["missing_caption_version"] += 1
                elif not isinstance(version, str) or not version.strip():
                    counts["invalid_caption_version"] += 1
                else:
                    versions[version] += 1
                caption = record.get("motion_caption")
                if isinstance(caption, str):
                    body = _body(caption)
                    counts["whole_caption_available_records"] += 1
                    scripts[_script_status(body)] += 1
                    _add_proxy(whole, body)
                else:
                    counts["invalid_motion_caption_records"] += 1
                raw_windows = record.get("windows")
                if (not isinstance(raw_windows, list) or not raw_windows
                        or any(not isinstance(window, dict) or not isinstance(window.get("caption"), str) for window in raw_windows)):
                    counts["structured_windows_unavailable_records"] += 1
                    if raw_windows is not None and raw_windows != []:
                        counts["malformed_structured_windows_records"] += 1
                else:
                    counts["structured_windows_available_records"] += 1
                    unique_in_windows = []
                    for window in raw_windows:
                        body = _body(window["caption"])
                        window_scripts[_script_status(body)] += 1
                        unique_in_windows.append(_add_proxy(windows, body))
                    unique_sum = sum(map(len, unique_in_windows))
                    counts["sum_window_unique_sentences"] += unique_sum
                    counts["cross_window_duplicate_sentence_excess"] += unique_sum - len(set().union(*unique_in_windows))
            after = _identity(os.fstat(stream.fileno()))
        if before != after or after != _identity(path.stat()) or size != after[2]:
            raise ValueError("Input changed during streaming audit: " + str(path))
        identities.append(after)
        inputs.append(dict(path=str(path), sha256=digest.hexdigest(), size_bytes=size, line_count=lines))
    for path, expected in zip(paths, identities):
        if _identity(path.stat()) != expected:
            raise ValueError("Input changed before audit completion: " + str(path))
    counts.update(unique_ids=len(ids), duplicate_distinct_ids=len(duplicate_ids), unique_source_groups=len(groups),
        ids_spanning_multiple_declared_splits=sum(len(splits) > 1 for splits in ids.values()),
        source_groups_spanning_multiple_declared_splits=sum(len(splits) > 1 for splits in groups.values()))
    return dict(schema="captionlab.label_audit.v1", audit_complete=True, quality_verified=False,
        training_approved=None, selected_training_records=None, inputs=inputs, counts=dict(counts),
        split_record_counts=dict(split_counts), caption_version_counts=dict(versions),
        body_character_status_counts=dict(scripts), structured_window_character_status_counts=dict(window_scripts),
        whole_caption_proxies=_finish_proxy(whole),
        structured_window_proxies=_finish_proxy(windows), error_line_examples=error_examples,
        definitions=dict(line_denominator="All physical lines, including blank or malformed lines; not a valid-label count.",
            metadata_denominator="Decoded JSON objects only; original nonempty string IDs/groups/splits are not normalized or aliased.",
            body="Remove explicit Dance style header lines and recognized leading numeric window labels only; sources are unchanged.",
            sentences="Split at period/semicolon followed by whitespace, or newline. Trim surrounding whitespace; preserve case and punctuation. Exact equality of these segments only.",
            words=_WORDS.pattern, action_word_vocabulary=sorted(_ACTION_WORDS),
            rep8="Lowercased English-word-regex 8-grams within each text; excess occurrences divided by available 8-grams; no evidence of erroneous dance repetition.",
            action_density="1000 * matched action words / regex word count; null for zero words. A lexical proxy, not narrative or semantic quality.",
            windows="Only explicit windows[*].caption strings; no timestamp parsing, inferred windows, duration or coverage verification.",
            cross_window_repetition="Sum of unique sentence counts per window minus unique sentences across windows, separately for each record.",
            character_status="Alphabetic characters whose Unicode names contain LATIN versus other alphabetic names only; Latin text is not verified English, and other scripts are not low quality.",
            split_conflicts="Shared declared IDs/source groups across literal split values; potential contamination signal, not verified song/artist independence."),
        limitations=["Audit completion and all lexical proxies do not approve labels, rewriting, visual grounding, event coverage or training suitability.",
                     "No source-before/output-after semantic comparison or fallback/validation provenance is inferred.",
                     "Real periodic actions and repetitions across windows remain valid possibilities; no automatic filtering or caption-version rewriting occurs."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError("Preserve previous audit output: " + str(output))
    if output in [path.resolve() for path in args.input]:
        raise ValueError("Output must not be an input")
    report = audit_labels(args.input)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(json.dumps(dict(audit_complete=True, quality_verified=False, training_approved=None,
                          line_count=report["counts"]["line_count"], output=str(output))))


if __name__ == "__main__":
    main()
