"""Frozen panel count and verified-cache reuse contracts, not pose accuracy."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

MS = Path(__file__).resolve().parents[1] / "scripts_omnidance/exp/ms"
sys.path.insert(0, str(MS))
import ms_body_panel as panel
from ms_optimize import sha256


def test_reuse_rejects_unmatched_source_or_model_protocol(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"synthetic-source-binding-only")
    row = {"id": "case", "source_group": "group", "video_path": str(video), "video_sha256": sha256(video)}
    source = {"id": "case", "source_group": "group", "source": "nlf_official_native_rgb_body",
              "source_sha256": {"video": row["video_sha256"], "model": "modelsha", "mapping": "mappingsha"},
              "num_aug": 3, "batch_size": 4, "point_count": 20, "raw_sha256": "rawsha", "frame_count": 5}
    raw = {"points_m": np.zeros((5, 20, 3))}
    record = {"id": "case", "status": "extracted", "bundle": str(tmp_path), "raw_sha256": "rawsha"}
    with mock.patch.object(panel, "load_bundle", return_value=(raw, source, "manifestsha")):
        assert panel.verify_case(record, row, "modelsha", "mappingsha", 3, 4) == record
        for model_sha, mapping_sha, aug, bs in (("wrong", "mappingsha", 3, 4),
                                                ("modelsha", "wrong", 3, 4),
                                                ("modelsha", "mappingsha", 1, 4),
                                                ("modelsha", "mappingsha", 3, 8)):
            with pytest.raises(ValueError):
                panel.verify_case(record, row, model_sha, mapping_sha, aug, bs)
        video.write_bytes(b"changed source")
        with pytest.raises(ValueError, match="RGB changed"):
            panel.verify_case(record, row, "modelsha", "mappingsha", 3, 4)


def test_failed_case_remains_failed_in_registry(tmp_path):
    failed = {"id": "case", "status": "failed", "error": "decode failure"}
    assert panel.verify_case(failed, {}, "", "", 3, 4) == failed


def test_wrong_frozen_panel_count_does_not_launch_inference(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    manifest = inputs / "manifest.jsonl"
    manifest.write_text("".join(json.dumps({"id": f"case{i}", "split": "train", "source_group": f"group{i}",
                                             "video_path": str(inputs / "video.mp4"), "panel_role": "development"}) + "\n"
                                 for i in range(19)))
    args = SimpleNamespace(out=str(tmp_path / "out"), manifest=str(manifest), model=str(inputs / "model"),
                           mapping=str(inputs / "mapping"), expected=20, workers=8, bs=4, num_aug=3, reuse=None)
    with mock.patch.object(panel.subprocess, "Popen") as process:
        with pytest.raises(ValueError, match="role/count/source"):
            panel.run(args)
        process.assert_not_called()
