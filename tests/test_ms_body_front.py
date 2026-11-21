"""Native body pipeline contracts using synthetic predictions, not accuracy tests."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

MS = Path(__file__).resolve().parents[1] / "scripts_omnidance/exp/ms"
sys.path.insert(0, str(MS))
import ms_body_front as front
from ms_landmarks import load_bundle
from ms_optimize import sha256
from motionscript_v2.selection import summarize_events
from motionscript_v2.body_events import extract_body_events, render_body_caption


class Predictor:
    point_count = 3
    method_schemas = {"detect_poses_batched": "synthetic"}
    spec = {"schema": "motionscript_landmark_spec_v1", "units": "m",
            "canonical_frame": "nlf_learned_native_joints",
            "points": [{"name": name, "canonical": [0., 0., 0.]}
                       for name in ("left_shoulder", "left_elbow", "left_wrist")],
            "source": {"method": "synthetic_test"}}

    def __init__(self, *args):
        self.position = 0

    def __call__(self, frames):
        result = {key: [] for key in ("boxes", "poses3d", "poses2d", "uncertainties")}
        for frame in frames:
            angle = np.deg2rad(120. - self.position * 10.)
            points = np.array([[0., -300., 3000.], [0., 0., 3000.],
                               [300. * np.sin(angle), -300. * np.cos(angle), 3000.]])
            result["boxes"].append(np.array([[0., 0., 100., 100., .9]]))
            result["poses3d"].append(points[None])
            result["poses2d"].append(np.full((1, 3, 2), 20.))
            result["uncertainties"].append(np.full((1, 3), .1))
            self.position += 1
        return result


def fixture(tmp_path, count=1):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    model = inputs / "model"
    model.write_bytes(b"synthetic-model-not-executable")
    mapping = inputs / "mapping.json"
    mapping.write_text("{}")
    video = inputs / "sample.mp4"
    video.write_bytes(b"synthetic-video-not-decodable")
    manifest = inputs / "manifest.jsonl"
    manifest.write_text("".join(json.dumps({"id": f"case{i}", "video_path": str(video), "split": "train",
                                            "source_group": f"group{i}", "video_sha256": sha256(video)}) + "\n"
                                for i in range(count)))
    args = SimpleNamespace(manifest=str(manifest), model=str(model), mapping=str(mapping),
                           out=str(tmp_path / "raw"), bs=3, num_aug=1, limit=0, device="cpu",
                           ffmpeg="stub", ffprobe="stub")
    info = {"time": np.arange(9) * .05, "fps": 20., "width": 100, "height": 100,
            "frame_indices": np.arange(9), "pts": np.arange(10, 19), "time_base": np.array([1, 20]),
            "pts_offset": .5, "media_end": .45, "media_end_source": "synthetic_last_frame_duration"}
    frames = np.zeros((9, 100, 100, 3), np.uint8)
    return args, info, [(0, frames[:3]), (3, frames[3:6]), (6, frames[6:])]


def test_native_body_extract_and_describe_preserve_pts_without_feet(tmp_path):
    args, info, batches = fixture(tmp_path)
    with mock.patch.object(front, "probe_video", return_value=info), mock.patch.object(front, "decode_batches", return_value=batches):
        terminal = front.extract(args, factory=Predictor)
    assert terminal["expected"] == 1
    assert terminal["records"][0]["status"] == "extracted"
    bundle = Path(terminal["records"][0]["bundle"])
    raw, manifest, _ = load_bundle(bundle)
    np.testing.assert_array_equal(raw["pts"], info["pts"])
    assert raw["points_m"].shape == (9, 3, 3)
    assert raw["valid"].all()
    assert not manifest["template_fk_used"]
    facts_args = SimpleNamespace(bundle=str(bundle), out=str(tmp_path / "facts"), ffprobe="stub",
        fps=20., max_gap_seconds=.2, smooth_seconds=0., max_point_uncertainty=None,
        min_duration_seconds=.15, min_angle_change_degrees=20., min_distance_change_m=.12)
    with mock.patch.object(front, "probe_video", return_value=info):
        report = front.describe(facts_args)
    bends = [event for event in report["events"] if event["body_part"] == "left_elbow"]
    assert len(bends) == 1 and bends[0]["action"] == "bends"
    assert bends[0]["evidence"]["source_frames"]["pts"][0] == 10
    assert report["quality"]["integer_pts_verified"]
    assert not report["human_review_completed"]
    assert not report["absence_of_events_means_static"]
    assert report["summary"]["selected_event_count"] == len(report["events"])


def test_failed_clips_remain_registered_in_denominator(tmp_path):
    args, info, batches = fixture(tmp_path, count=2)
    with mock.patch.object(front, "probe_video", side_effect=[info, ValueError("synthetic decoding failure")]), mock.patch.object(front, "decode_batches", return_value=batches):
        terminal = front.extract(args, factory=Predictor)
    assert terminal["expected"] == 2 and len(terminal["records"]) == 2
    assert terminal["records"][1]["status"] == "failed"
    assert terminal["failures_in_denominator"]
    assert not terminal["semantic_approval"]


def test_native_alias_contract_rejects_wrong_anatomy_and_toe_substitutions():
    joints = [{"name": name, "model_name": model_name} for model_name, name in front.NATIVE_ALIASES.items()]
    mapping = {"schema": "nlf_native_body_mapping_v1", "mapping_version": "smpl24-body20-no-toes-v1",
               "skeleton": "smpl_24", "joints": joints}
    front.validate_native_aliases(mapping)
    joints[0]["name"] = "left_wrist"
    with pytest.raises(ValueError, match="Anatomical alias"):
        front.validate_native_aliases(mapping)
    joints[0] = {"name": "left_foot", "model_name": "ltoe"}
    with pytest.raises(ValueError, match="toe"):
        front.validate_native_aliases(mapping)


def test_native_query_count_is_not_fixed_to_32():
    result = Predictor()(np.zeros((2, 4, 4, 3), np.uint8))
    selected = front.select_largest_detections(result, 2, 3)
    assert selected["points_m"].shape == (2, 3, 3)
    with pytest.raises(ValueError):
        front.select_largest_detections(result, 2, 32)


def test_repetition_summary_accepts_independent_point_segments():
    angles = np.r_[np.linspace(40, 120, 9), np.linspace(120, 40, 9)[1:],
                   np.linspace(40, 120, 9)[1:], np.linspace(120, 40, 9)[1:]]
    positions = np.zeros((len(angles), 3, 3))
    positions[:, 0, 1] = -.3
    positions[:, 2, 0] = .3 * np.sin(np.deg2rad(angles))
    positions[:, 2, 1] = -.3 * np.cos(np.deg2rad(angles))
    data = {"time": np.arange(len(angles)) / 20, "points_m": positions,
            "names": ["left_shoulder", "left_elbow", "left_wrist"],
            "point_valid": np.ones((len(angles), 3), bool), "point_observed": np.ones((len(angles), 3), bool),
            "point_interpolated": np.zeros((len(angles), 3), bool),
            "point_segment_id": np.zeros((len(angles), 3), int),
            "coordinate_frame": "camera_xright_ydown_zforward", "units": "m"}
    events = [event for event in extract_body_events(data) if event["body_part"] == "left_elbow"]
    summary = summarize_events(events, renderer=render_body_caption)
    assert len(events) == 4
    assert len(summary["groups"]) == 1
    assert len(summary["groups"][0]["occurrences"]) == 4
    assert summary["groups"][0]["kind"] == "repeated_alternation"
    assert summary["source_event_count"] == summary["selected_event_count"] == 4
