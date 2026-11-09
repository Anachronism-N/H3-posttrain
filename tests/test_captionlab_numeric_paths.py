"""Numeric evidence IDs are filenames, never paths or case-insensitive aliases."""
import json

import pytest

pytest.importorskip("torch")
from captionlab.numeric_common import rows_from_manifest


def manifest(tmp_path, identifiers, *, raw_unicode=False):
    cases = []
    for index, identifier in enumerate(identifiers):
        cases.append({"case_id": "public-case-" + str(index), "target_source": {
            "id": identifier, "source_group_id": "source-" + str(index),
            "audio": {"path": "unopened.wav", "sha256": "a" * 64,
                      "sample_rate": 16000, "num_samples": 81000},
            "raw_caption": "Step left."}, "target_windows": [{"start": 0.0, "end": 5.0625}]})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"cases": cases}, ensure_ascii=not raw_unicode), encoding="utf-8")
    return path


@pytest.mark.parametrize("identifier", [
    "case-a_123", "case.with.dots", "A B", "leading-space allowed", "0", ".hidden-case",
    "COM0", "COM10", "COM1-not-a-device", "container", "nul-extra", "auxiliary", "100%2fcase",
    "舞步-片段０１", "café_ß", "动作-É", "춤-01", "رقصة-01", "event-😀",
])
def test_ordinary_unicode_ids_remain_unchanged(identifier, tmp_path):
    path = manifest(tmp_path, [identifier, "second-fixed-case", "third-fixed-case"], raw_unicode=True)
    raw = path.read_bytes()
    rows = rows_from_manifest(path)
    assert rows[0]["id"] == identifier
    assert rows[0]["audio_path"] == "unopened.wav"
    assert rows[0]["target_text"] == "Step left."
    assert path.read_bytes() == raw


@pytest.mark.parametrize("identifier", [
    "../escape", "..\\escape", "/absolute", "\\absolute", "C:\\escape", "C:relative",
    "\\\\server\\share", "folder/name", "folder\\name", "colon:name", 'quote"name',
    "left<right", "left>right", "left|right", "wild*card", "wild?card", ".", "..", "...",
    "trailing.", "trailing ", "trailing. ", "trailing\u00a0", "trailing\u3000",
    "", " ", None, True, 17, ["name"], {"name": "id"}, "unpaired-\ud800",
])
def test_nonportable_or_path_ids_rejected_explicitly(identifier, tmp_path):
    path = manifest(tmp_path, ["first", identifier, "third"])
    with pytest.raises(ValueError, match="Numeric source id"):
        rows_from_manifest(path)


@pytest.mark.parametrize("control", list(range(32)) + [127, 128, 133, 159])
def test_control_characters_rejected(control, tmp_path):
    path = manifest(tmp_path, ["first", "id" + chr(control) + "rest", "third"])
    with pytest.raises(ValueError, match="control character"):
        rows_from_manifest(path)


@pytest.mark.parametrize("identifier", [
    "CON", "con", "Con.txt", "PRN", "prn.log", "AUX", "aux.json", "NUL", "nul.pt",
    "COM1", "com9", "COM3.txt", "LPT1", "lpt9", "lpt4.log", "CONIN$", "conout$.pt", "CLOCK$",
    "COM¹", "com².pt", "COM³", "LPT¹", "lpt².log", "LPT³", "con .txt", "aux...txt",
])
def test_windows_device_names_rejected_even_on_non_windows(identifier, tmp_path):
    path = manifest(tmp_path, ["first", identifier, "third"])
    with pytest.raises(ValueError, match="reserved Windows device name"):
        rows_from_manifest(path)


@pytest.mark.parametrize("identifiers", [
    ["case", "CASE", "third"], ["case", "case", "third"], ["é", "É", "third"],
    ["Straße", "STRASSE", "third"], ["第一首", "第二首", "第一首"],
])
def test_case_insensitive_duplicate_ids_rejected_without_renaming(identifiers, tmp_path):
    path = manifest(tmp_path, identifiers)
    with pytest.raises(ValueError, match="Duplicate numeric source id.*case-insensitive"):
        rows_from_manifest(path)


@pytest.mark.parametrize("identifiers", [["first", "second", "../escape"], ["first", "second", "FIRST"]])
def test_entire_manifest_checked_before_a_consumer_can_write_first_case(identifiers, tmp_path):
    path = manifest(tmp_path, identifiers)
    output = tmp_path / "evidence"; output.mkdir()
    with pytest.raises(ValueError):
        for row in rows_from_manifest(path):
            (output / (row["id"] + "-numeric.pt")).write_bytes(b"CPU fixture, no model")
    assert not list(output.iterdir())
    assert not (tmp_path / "escape-numeric.pt").exists()


@pytest.mark.parametrize("identifiers", [[], ["one"], ["one", "two"], ["one", "two", "three", "four"]])
def test_three_case_numeric_scope_is_preserved(identifiers, tmp_path):
    with pytest.raises(ValueError, match="Exactly three fixed source cases"):
        rows_from_manifest(manifest(tmp_path, identifiers))
