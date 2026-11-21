"""Postprocessing registration and failure-denominator contracts."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

MS = Path(__file__).resolve().parents[1] / "scripts_omnidance/exp/ms"
sys.path.insert(0, str(MS))
import ms_body_postprocess as postprocess


def args(tmp_path, registry):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    path = inputs / "registry.json"
    path.write_text(json.dumps(registry))
    return SimpleNamespace(registry=str(path), out=str(tmp_path / "output"), workers=2,
                           python=sys.executable, ffmpeg="unused", ffprobe="unused",
                           reuse_facts=None, reuse_evidence=None, protocol=None)


def test_failed_source_is_not_dropped_or_marked_as_quality_success(tmp_path):
    record = {"id": "case", "status": "failed", "error": "source decode failed"}
    result = postprocess.run(args(tmp_path, {"expected": 1, "records": [record]}))
    assert result["expected"] == 1 and len(result["records"]) == 1
    assert result["records"][0]["status"] == "source_extraction_failed"
    assert result["failures_in_denominator"]
    assert not result["all_candidates_have_evidence"]
    assert not result["quality_gate_passed"]


def test_acceptance_without_frozen_protocol_is_rejected(tmp_path):
    records = [{"id": f"case{i}", "status": "failed"} for i in range(200)]
    with pytest.raises(ValueError, match="Frozen measurement protocol"):
        postprocess.run(args(tmp_path, {"expected": 200, "records": records}))


def test_registry_count_mismatch_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="count/IDs"):
        postprocess.run(args(tmp_path, {"expected": 20, "records": []}))
