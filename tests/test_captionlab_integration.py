"""Synthetic end-to-end probes; results are software tests, never quality evidence."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from captionlab.io import file_sha, loads, object_sha, read_json, write_json
from captionlab.manifest import prepare_probe, source_components
from captionlab.runner import audit_runtime, run_probe, validate_probe, _load_runtime, _source_row
from captionlab.scoring import summarize_scores
from captionlab.__main__ import main


def source(identifier, root, split="val"):
    rate, frames = 8000, 40000
    wave = (.05 * np.sin(np.arange(frames) * (0.1 + ord(identifier[0]) / 1000))).astype("float32")
    path = root / (identifier + ".wav")
    sf.write(path, wave, rate, subtype="FLOAT")
    return {"schema": "captionlab.source.v1", "id": identifier, "split": split,
            "source_group_id": "video-" + identifier, "song_id": "song-" + identifier,
            "near_duplicate_group_id": None,
            "raw_caption": "Dance style: contemporary.\n[0-2s] One side wrist moves left.\n[2-5s] The other side knee bends.\n",
            "genre_header_template": "Dance style: {genre}.\n",
            "audio": {"path": str(path), "sha256": file_sha(path),
                      "pcm_sha256": hashlib.sha256(wave.tobytes()).hexdigest(),
                      "sample_rate": rate, "num_samples": frames,
                      "source_start_seconds": 0, "pad_intervals": []},
            "windows": [{"start": 0, "end": 2, "context_dependency": "independent"},
                        {"start": 2, "end": 5, "context_dependency": "dependent"}]}


@pytest.fixture
def rows(tmp_path):
    return [source(x, tmp_path) for x in ("a", "b", "c", "d")]


def prepare(rows, **kwargs):
    return prepare_probe(rows, genre_classes=["contemporary", "ballet", "unknown"], **kwargs)


def hard(case="a", donor="c", review="unreviewed"):
    return {"case_id": case, "donor_id": donor, "mismatch_review": review,
            "review_evidence": "fixture only" if review == "reviewed_mismatch" else ""}


def test_preparation_roundtrip_to_validator_and_missing_summary(rows):
    for pairs in ([], [hard()]):
        manifest = prepare(rows, case_ids=["a"], hard_pairs=pairs)
        validate_probe(manifest)
        summary = summarize_scores(manifest, [], n_resamples=2)
        assert summary["coverage"]["n_cases_expected"] == 1
        assert manifest["forward_calls_per_checkpoint"] == (4 if pairs else 3)
        assert [r["audio_id"] for r in manifest["cases"][0]["conditions"] if r["kind"] == "true"] == ["a"]


def test_shuffle_invariant_and_no_target_mutation(rows):
    original = copy.deepcopy(rows)
    left = prepare(rows, seed=2, max_cases=2)
    right = prepare(list(reversed(rows)), seed=2, max_cases=2)
    assert left["cases"] == right["cases"]
    assert rows == original
    assert all(c["target_text"] == c["target_source"]["raw_caption"] for c in left["cases"])


def test_known_positive_and_transitive_groups_never_negatives(rows):
    rows[0]["song_id"] = rows[1]["song_id"]
    rows[1]["near_duplicate_group_id"] = rows[2]["near_duplicate_group_id"] = "near-shared"
    groups = source_components(rows)
    assert groups["a"] == groups["c"] != groups["d"]
    manifest = prepare(rows, case_ids=["a"])
    assert next(c for c in manifest["cases"][0]["conditions"] if c["kind"] == "random")["audio_id"] == "d"
    with pytest.raises(ValueError, match="known positive"):
        prepare(rows, case_ids=["a"], hard_pairs=[hard()])


@pytest.mark.parametrize("shared", ["sha256", "path", "pcm_sha256"])
def test_same_file_or_pcm_excluded_even_if_other_metadata_conflicts(rows, shared):
    rows[2]["audio"][shared] = rows[0]["audio"][shared]
    with pytest.raises(ValueError, match="known positive"):
        prepare(rows, case_ids=["a"], hard_pairs=[hard()])


@pytest.mark.parametrize("field,value", [("sample_rate", 16000), ("num_samples", 80000), ("pad_intervals", [[0, 1]])])
def test_geometry_mismatch_not_a_negative_shortcut(rows, field, value):
    rows[2]["audio"][field] = value
    if field == "sample_rate":
        rows[2]["audio"]["num_samples"] *= 2
    with pytest.raises(ValueError, match="geometry"):
        prepare(rows, case_ids=["a"], hard_pairs=[hard()])
    result = prepare(rows, case_ids=["a"])
    assert all(c["audio_id"] != "c" for c in result["cases"][0]["conditions"])


def test_all_related_leaves_explicit_unavailable_conditions(rows):
    for row in rows:
        row["song_id"] = "same-song"
    manifest = prepare(rows, case_ids=["a"])
    validate_probe(manifest)
    assert manifest["forward_calls_per_checkpoint"] == 2
    assert {c["kind"] for c in manifest["cases"][0]["unavailable_conditions"]} == {"random", "hard"}
    result = summarize_scores(manifest, [], n_resamples=2)
    assert result["coverage"]["n_conditions_unavailable"] == 2


def test_never_select_test_or_train_and_cross_split_rejected(rows):
    rows[1]["split"] = "test"
    with pytest.raises(ValueError, match="non-validation"):
        prepare(rows, case_ids=["b"])
    with pytest.raises(ValueError, match="crosses split"):
        prepare(rows, case_ids=["a"], hard_pairs=[hard(donor="b")])
    rows[1]["song_id"] = rows[0]["song_id"]
    with pytest.raises(ValueError):
        prepare(rows)


def test_manifest_cannot_silently_edit_target_transform_or_audio(rows):
    original = prepare(rows, case_ids=["a"])
    for field, value in (("target_text", "Other caption"), ("target_mode", "actions_untimed")):
        bad = copy.deepcopy(original)
        bad["cases"][0][field] = value
        with pytest.raises(ValueError):
            validate_probe(bad)
    bad = copy.deepcopy(original)
    bad["cases"][0]["conditions"][-1]["transform"] = "identity"
    with pytest.raises(ValueError, match="transform"):
        validate_probe(bad)
    bad = copy.deepcopy(original)
    bad["cases"][0]["target_windows"][0]["caption"] = "invented action"
    with pytest.raises(ValueError, match="original action spans"):
        validate_probe(bad)


def test_cli_prepare_and_targets_run_without_torch(rows, tmp_path):
    src = tmp_path / "sources.json"
    write_json(src, {"schema": "captionlab.sources.v1", "genre_classes": ["contemporary"], "records": rows})
    output = tmp_path / "probe.json"
    assert main(["prepare", "--sources", str(src), "--out", str(output), "--max-cases", "2"]) == 0
    validate_probe(read_json(output))
    targets, crops = tmp_path / "targets.json", tmp_path / "crops.json"
    main(["targets", "--sources", str(src), "--out", str(targets), "--crop-out", str(crops)])
    assert len(read_json(targets)["records"]) == 4
    assert len(read_json(crops)["windows"]) == 8
    with pytest.raises(FileExistsError):
        main(["prepare", "--sources", str(src), "--out", str(output)])


def runtime_config(tmp_path, factory_text):
    factory = tmp_path / "factory.py"
    factory.write_text(factory_text, encoding="utf-8")
    artifacts = []
    for role in ("model", "tokenizer", "runtime"):
        path = tmp_path / (role + ".txt")
        path.write_text("test fixture: " + role, encoding="utf-8")
        artifacts.append({"role": role, "path": str(path), "sha256": file_sha(path)})
    return {"schema": "captionlab.runtime.v1", "protocol_id": "fixture-v1", "target_mode": "raw",
            "checkpoint_mode": "base_only", "factory": {"path": str(factory), "sha256": file_sha(factory)},
            "artifacts": artifacts}


FACTORY = '''
def load_runtime():
    import torch
    from types import SimpleNamespace
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
        def forward(self, input_ids, attention_mask, use_cache):
            assert not use_cache and not torch.is_grad_enabled() and not self.training
            return SimpleNamespace(logits=torch.zeros(1, input_ids.shape[1], 258))
    class Adapter:
        max_length = 8192
        def __init__(self): self.model = Model()
        def eos_token_ids(self): return [257]
        def collate(self, rows):
            assert rows[0]['audio_duration_seconds'] == 5.0
            assert rows[0]['target_windows'][0]['caption'] == 'One side wrist moves left.'
            ids = [b+1 for b in rows[0]['target_text'].encode('utf-8')] + [257]
            x = torch.tensor([[0]+ids])
            return dict(input_ids=x, attention_mask=torch.ones_like(x), labels=torch.tensor([[-100]+ids]))
        def _move(self, batch): return {k:v for k,v in batch.items() if k != 'labels'}
    return dict(adapter=Adapter(), receipt=dict(protocol_id='fixture-v1', checkpoint_mode='base_only',
                target_contract_verified=True, fixture_only=True))
'''


def test_full_cpu_teacher_forcing_run_and_exclusive_outputs(rows, tmp_path):
    pytest.importorskip("torch")
    manifest = prepare(rows, case_ids=["a"], hard_pairs=[hard()])
    manifest_path, config_path = tmp_path / "probe.json", tmp_path / "runtime.json"
    write_json(manifest_path, manifest)
    write_json(config_path, runtime_config(tmp_path, FACTORY))
    out = tmp_path / "run"
    result = run_probe(manifest_path, config_path, out)
    assert result["new_training_updates"] == result["new_generation_calls"] == 0
    records = [loads(line) for line in (out / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4
    assert len({tuple(r["target_token_ids"]) for r in records}) == 1
    for record in records:
        consumed = record["runtime_details"]["audio"]
        assert Path(consumed["path"]).parent == out
        assert consumed["sha256"] == file_sha(consumed["path"])
        if record["condition_id"] == "silence":
            data, rate = sf.read(consumed["path"])
            assert len(data) == 40000 and rate == 8000 and np.count_nonzero(data) == 0
    with pytest.raises(FileExistsError):
        run_probe(manifest_path, config_path, out)
    main(["summarize", "--manifest", str(manifest_path), "--scores", str(out / "scores.jsonl"),
          "--out", str(tmp_path / "resummary.json"), "--resamples", "10"])


def test_audited_factory_does_not_execute_stale_same_mtime_pyc(tmp_path):
    prefix = "def load_runtime():\n    return {'adapter': '%s', 'receipt': {'protocol_id':'fixture-v1', 'checkpoint_mode':'base_only', 'target_contract_verified':True}}\n"
    config = runtime_config(tmp_path, prefix % "OLD")
    path = Path(config["factory"]["path"])
    import py_compile
    py_compile.compile(str(path), doraise=True)
    stat = path.stat()
    path.write_text(prefix % "NEW", encoding="utf-8")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    config["factory"]["sha256"] = file_sha(path)
    assert _load_runtime(config, audit_runtime(config, tmp_path))["adapter"] == "NEW"


def test_snapshot_survives_source_mutation_after_audited_read(rows, tmp_path):
    manifest = prepare(rows, case_ids=["a"])
    case = manifest["cases"][0]
    cond = case["conditions"][0]
    out = tmp_path / "snapshots"
    out.mkdir()
    cache = {}
    row = _source_row(case, cond, tmp_path, out, cache)
    Path(cond["audio"]["path"]).write_bytes(b"changed external source")
    assert file_sha(row["audio_path"]) == row["audio_sha256"] == cond["audio"]["sha256"]
    reused = _source_row(case, cond, tmp_path, out, cache)
    assert reused == row


def test_runtime_audit_and_failure_preserve_denominator(rows, tmp_path):
    config = runtime_config(tmp_path, "def load_runtime():\n    raise RuntimeError('fixture load failure')\n")
    manifest_path, config_path = tmp_path / "probe.json", tmp_path / "runtime.json"
    write_json(manifest_path, prepare(rows, case_ids=["a"]))
    write_json(config_path, config)
    out = tmp_path / "failed-run"
    with pytest.raises(RuntimeError, match="fixture load failure"):
        run_probe(manifest_path, config_path, out)
    assert read_json(out / "failure.json")["completed_records"] == 0
    assert read_json(out / "failure_summary.json")["run_status"] == "failed"
    assert len(read_json(out / "manifest.json")["cases"]) == 1
    Path(config["artifacts"][0]["path"]).write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        audit_runtime(config, tmp_path)


def test_json_rejects_nan_and_duplicate_keys():
    for invalid in ('{"x":NaN}', '{"x":1,"x":2}'):
        with pytest.raises(ValueError):
            loads(invalid)
