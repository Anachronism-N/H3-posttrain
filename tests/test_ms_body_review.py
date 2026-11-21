"""Timing, projection and source-binding contracts for evidence export."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

MS = Path(__file__).resolve().parents[1] / "scripts_omnidance/exp/ms"
sys.path.insert(0, str(MS))
import ms_body_front as front
import ms_body_review as review
from ms_optimize import sha256
from test_ms_body_front import Predictor


def test_source_interpolation_keeps_point_gaps_and_does_not_extrapolate():
    tracks = {"time": np.array([.2, .3, .5]), "points_m": np.arange(18, dtype=float).reshape(3, 2, 3),
              "point_valid": np.ones((3, 2), bool), "point_segment_id": np.array([[0, 0], [0, 1], [0, 1]])}
    points, valid = review.at_source_times(tracks, np.array([.1, .2, .25, .4, .5, .6]))
    assert not valid[[0, 5]].any()
    assert valid[2].tolist() == [True, False]
    assert valid[3].all()
    np.testing.assert_allclose(points[2, 0], [3, 4, 5])
    assert np.isnan(points[2, 1]).all()


def test_projection_uses_official_camera_axes_not_body_y_up():
    points = np.array([[[0., 0., 3.], [.1, .2, 3.], [0., 0., -1.], [1., 1., 0.]]])
    pixels, good = review.project_camera(points, 100, 80)
    np.testing.assert_allclose(pixels[0, 0], [49.5, 39.5])
    assert pixels[0, 1, 0] > 49.5 and pixels[0, 1, 1] > 39.5
    assert good.tolist() == [[True, True, False, False]]
    assert np.isnan(pixels[0, 2:]).all()


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="Verified FFmpeg/FFprobe required")
def test_generated_video_evidence_preserves_every_frame_and_template_is_unreviewed(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    video = inputs / "source.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=96x128:rate=20", "-frames:v", "9", "-threads", "1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True)
    model = inputs / "model"
    model.write_bytes(b"synthetic-not-executable")
    mapping = inputs / "mapping.json"
    mapping.write_text("{}")
    manifest = inputs / "manifest.jsonl"
    manifest.write_text(json.dumps({"id": "fixture", "split": "train", "source_group": "fixture_group",
                                    "video_path": str(video), "video_sha256": sha256(video)}) + "\n")
    extract_args = SimpleNamespace(manifest=str(manifest), model=str(model), mapping=str(mapping),
        out=str(tmp_path / "raw"), bs=3, num_aug=1, limit=0, device="cpu", ffmpeg="ffmpeg", ffprobe="ffprobe")
    front.extract(extract_args, factory=Predictor)
    bundle = tmp_path / "raw" / "fixture"
    facts_args = SimpleNamespace(bundle=str(bundle), out=str(tmp_path / "facts"), ffprobe="ffprobe",
        fps=20., max_gap_seconds=.2, smooth_seconds=0., max_point_uncertainty=None,
        min_duration_seconds=.15, min_angle_change_degrees=20., min_distance_change_m=.12)
    front.describe(facts_args)
    args = SimpleNamespace(bundle=str(bundle), facts=str(tmp_path / "facts"), out=str(tmp_path / "evidence"),
                           ffmpeg="ffmpeg", ffprobe="ffprobe")
    report = review.export(args)
    assert report["frame_count"] == 9
    assert [frame["frame_index"] for frame in report["frames"]] == list(range(9))
    assert all(frame["image"].endswith(".png") for frame in report["frames"])
    template = json.loads((tmp_path / "evidence" / "review_template.json").read_text())
    assert not template["review_completed"]
    assert not template["reference_built_from_rgb_before_reading_candidates"]
    assert not report["semantic_approval"] and not report["human_review_completed"]
    assert report["rendered_final_frame_duration_is_not_source_evidence"]
    with pytest.raises(ValueError, match="Facts receipt"):
        (tmp_path / "facts" / "facts.json").write_bytes(b"{}")
        review.load_facts(tmp_path / "facts")
