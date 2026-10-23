"""Chunk/media-time contracts; no beat detector or descriptor model is run."""

import copy
import json

import numpy as np
import pytest

from scripts_omnidance.exp.ms.motionscript_v2.chunks import (
    SCHEMA,
    fixed_windows,
    slice_landmarks,
    validate_windows,
)


def spec(windows=None, *, strategy="explicit", **extra):
    return {"schema": SCHEMA, "strategy": strategy,
            "windows": [{"start": 0, "end": 2}, {"start": 2, "end": 5.0625}] if windows is None else windows,
            **extra}


def beat_spec():
    return spec([{"start": 0, "end": 1.72}, {"start": 1.72, "end": 3.47},
                 {"start": 3.47, "end": 5.0625}], strategy="beat",
                beats=[.13, .61, 1.18, 1.72, 2.27, 2.89, 3.47, 4.05, 4.7],
                beat_source={"method": "external_actual_beat_times", "version": "synthetic-test",
                             "reliability": {"status": "unverified"}})


def tracks(n=11, *, start=0, period=.05):
    # All leading lengths intentionally match. Names and raw tracks must still
    # remain intact when only a few target timestamps are selected.
    time = start + np.arange(n, dtype=float) * period
    points = np.broadcast_to(time[:, None, None], (n, n, 3)).copy().astype(np.float32)
    return {"time": time, "points_m": points,
            "point_valid": np.ones((n, n), bool),
            "point_observed": np.ones((n, n), bool),
            "point_interpolated": np.zeros((n, n), bool),
            "point_segment_id": np.full((n, n), 7, dtype=np.int32),
            "names": np.asarray([f"point_{i}" for i in range(n)]),
            "coordinate_frame": np.asarray("camera_xright_ydown_zforward"), "units": np.asarray("m"),
            "raw_time": np.arange(n, dtype=float) / 16,
            "raw_points_m": points.copy(), "raw_valid": np.ones((n, n), bool),
            "raw_uncertainty": np.zeros((n, n), np.float32),
            "raw_points2d_px": np.zeros((n, n, 2), np.float32),
            "landmark_signal_values": np.arange(n * n, dtype=float).reshape(n, n),
            "landmark_signal_valid": np.ones((n, n), bool),
            "landmark_signal_names": np.asarray([f"signal_{i}" for i in range(n)]),
            "other_static_metadata": np.eye(n)}


def test_fixed_windows_merge_documented_short_tail_and_keep_global_origin():
    assert fixed_windows(0, 5.0625) == [{"start": 0., "end": 2.}, {"start": 2., "end": 5.0625}]
    assert fixed_windows(1.25, 6.3125, 2., 1.5) == [
        {"start": 1.25, "end": 3.25}, {"start": 3.25, "end": 6.3125}]
    assert fixed_windows(0, 1.1) == [{"start": 0., "end": 1.1}]
    assert fixed_windows(0, 4) == [{"start": 0., "end": 2.}, {"start": 2., "end": 4.}]


def test_tail_at_threshold_is_retained_and_zero_disables_merging():
    assert fixed_windows(0, 5.5)[-1] == {"start": 4., "end": 5.5}
    assert fixed_windows(0, 5.0625, 2, 0)[-1] == {"start": 4., "end": 5.0625}
    assert len(fixed_windows(0, 5.0625, 2, 0)) == 3


@pytest.mark.parametrize("arguments", [(0, 0), (2, 1), (-1, 4), (0, np.nan),
                                       (0, 4, 0), (0, 4, True), (0, 4, np.inf),
                                       (0, 4, 2, -1), (0, 4, 2, np.nan)])
def test_invalid_fixed_parameters_fail(arguments):
    with pytest.raises(ValueError):
        fixed_windows(*arguments)


def test_validate_preserves_content_binding_and_declared_provenance():
    source = spec(strategy="fixed", video_sha256="A" * 64,
                  source={"caption_source": "nlf_direct_rgb_landmarks", "source_group": "video_17"})
    source["windows"][0]["note"] = "input window, not action bounds"
    original = copy.deepcopy(source)
    normalized = validate_windows(source, media_start=0, media_end=5.0625)
    assert normalized["source"] == source["source"]
    assert normalized["video_sha256"] == "a" * 64
    assert normalized["windows"][0]["note"] == source["windows"][0]["note"]
    assert normalized["media_start"] == 0 and normalized["media_end"] == 5.0625
    assert normalized["validation"]["source_provenance_verified"] is False
    assert normalized["validation"]["window_boundaries_are_action_boundaries"] is False
    assert source == original
    json.dumps(normalized, allow_nan=False)


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(schema="wrong"),
    lambda value: value.update(strategy="bpm"),
    lambda value: value.update(windows=[]),
    lambda value: value["windows"][0].update(start=.1),
    lambda value: value["windows"][0].update(end=1.9),
    lambda value: value["windows"][0].update(end=2.1),
    lambda value: value["windows"][1].update(end=5.),
    lambda value: value["windows"][1].update(end=6.),
    lambda value: value["windows"][0].update(start=True),
    lambda value: value["windows"][0].update(end=0),
    lambda value: value.update(media_start=.1),
    lambda value: value.update(media_end=5.),
    lambda value: value.update(video_sha256="not-a-hash"),
    lambda value: value.update(source={"uncertainty": np.nan}),
])
def test_incomplete_overlapping_or_invalid_window_spec_is_rejected(mutation):
    source = spec()
    mutation(source)
    with pytest.raises(ValueError):
        validate_windows(source, media_start=0, media_end=5.0625)


def test_nonuniform_actual_beats_align_only_interior_boundaries():
    normalized = validate_windows(beat_spec(), media_start=0, media_end=5.0625)
    assert normalized["windows"][0]["start"] == 0  # Not itself a beat.
    assert normalized["windows"][-1]["end"] == 5.0625  # Not itself a beat.
    assert [window["end"] for window in normalized["windows"][:-1]] == [1.72, 3.47]
    assert normalized["validation"]["beat_boundary_alignment_checked"] is True
    assert normalized["validation"]["beats_verified_against_audio"] is False
    assert normalized["beat_source"]["reliability"] == {"status": "unverified"}


def test_regular_actual_beat_list_is_not_rejected_for_being_regular():
    source = spec(strategy="beat", beats=[0, .5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.],
                  beat_source={"source": "externally_annotated_beats", "reliability": "unverified"})
    normalized = validate_windows(source, media_start=0, media_end=5.0625)
    assert normalized["windows"] == source["windows"]


@pytest.mark.parametrize("mutation", [
    lambda value: value.pop("beats"),
    lambda value: value.update(beats=[]),
    lambda value: value.update(beats=[.1, .1, 1.72, 3.47]),
    lambda value: value.update(beats=[.1, 1.72, 1.2, 3.47]),
    lambda value: value.update(beats=[-.1, 1.72, 3.47]),
    lambda value: value.update(beats=[.1, 1.72, 3.47, 6]),
    lambda value: value.update(beats=[.1, 1.72, np.inf]),
    lambda value: value["windows"][0].update(end=2.),
    lambda value: value.pop("beat_source"),
    lambda value: value.update(beat_source={"reliability": "unknown"}),
    lambda value: value.update(beat_source={"method": "BeatThis"}),
    lambda value: value.update(beat_source={"method": "global_bpm_grid", "reliability": "unknown"}),
    lambda value: value.update(beat_source={"method": "external", "reliability": "unknown", "derived_from_global_bpm": True}),
])
def test_invalid_or_bpm_synthesized_beat_specs_fail(mutation):
    source = beat_spec()
    mutation(source)
    with pytest.raises(ValueError):
        validate_windows(source, media_start=0, media_end=5.0625)


def test_numerical_boundary_normalization_does_not_create_gaps():
    source = spec([{"start": 1e-12, "end": 1.72 + 1e-12},
                   {"start": 1.72, "end": 3.47}, {"start": 3.47, "end": 5.0625 - 1e-12}],
                  strategy="beat", beats=beat_spec()["beats"], beat_source=beat_spec()["beat_source"])
    normalized = validate_windows(source, media_start=0, media_end=5.0625)
    assert normalized["windows"][0] == {"start": 0., "end": 1.72}
    assert normalized["windows"][1]["start"] == normalized["windows"][0]["end"]
    assert normalized["windows"][-1]["end"] == 5.0625


def test_slicing_is_half_open_and_does_not_duplicate_boundary_samples():
    source = tracks()
    windows = fixed_windows(0, .55, .2, 0)
    selected = []
    for window in windows:
        result, metadata = slice_landmarks(source, window["start"], window["end"])
        assert np.all(result["time"] >= window["start"])
        assert np.all(result["time"] < window["end"])
        selected.extend(metadata["selected_indices"])
    assert selected == list(range(11))
    assert len(set(selected)) == len(selected)


def test_slicing_preserves_raw_arrays_names_signal_names_and_static_metadata():
    source = tracks()
    result, metadata = slice_landmarks(source, .1, .3)
    expected = np.flatnonzero((source["time"] >= .1) & (source["time"] < .3))
    assert result["points_m"].shape == (len(expected), 11, 3)
    assert result["landmark_signal_values"].shape == (len(expected), 11)
    np.testing.assert_array_equal(result["landmark_signal_values"], source["landmark_signal_values"][expected])
    np.testing.assert_array_equal(result["landmark_signal_valid"], source["landmark_signal_valid"][expected])
    for key in ("raw_time", "raw_points_m", "raw_valid", "raw_uncertainty", "raw_points2d_px",
                "names", "landmark_signal_names", "other_static_metadata", "coordinate_frame", "units"):
        np.testing.assert_array_equal(result[key], source[key])
        assert key not in metadata["sliced_time_fields"]
    assert metadata["raw_arrays_retained_full_length"] is True


def test_global_time_geometry_and_segment_ids_are_never_reset_or_stitched():
    source = tracks(start=2., period=.25)
    source["point_valid"][4, 1] = False
    source["points_m"][4, 1] = np.nan
    source["point_segment_id"][4, 1] = -1
    source["point_segment_id"][5:, 1] = 99
    result, metadata = slice_landmarks(source, 2.5, 4.)
    np.testing.assert_array_equal(result["time"], source["time"][2:8])
    np.testing.assert_allclose(result["points_m"], source["points_m"][2:8], equal_nan=True)
    assert result["point_segment_id"][:, 1].tolist() == [7, 7, -1, 99, 99, 99]
    assert not result["point_valid"][2, 1]
    assert metadata["coordinate_reset_applied"] is False
    assert metadata["segment_ids_preserved"] is True
    assert metadata["first_sample_time"] == 2.5
    assert metadata["last_sample_time"] == 3.75
    assert metadata["window_end"] == 4.
    assert metadata["sample_span_is_complete_window_evidence"] is False
    assert metadata["selected_indices"] == list(range(2, 8))


def test_empty_window_abstains_without_filling_or_claiming_static_motion():
    source = tracks()
    result, metadata = slice_landmarks(source, 1., 2.)
    assert result["time"].shape == (0,)
    assert result["points_m"].shape == (0, 11, 3)
    assert result["landmark_signal_values"].shape == (0, 11)
    assert result["names"].shape == (11,)
    assert result["raw_time"].shape == (11,)
    assert metadata["empty"] and metadata["abstain"]
    assert metadata["abstain_reason"] == "no_samples_in_window"
    assert metadata["first_sample_time"] is metadata["last_sample_time"] is None
    assert metadata["point_support_fraction"] == 0
    json.dumps(metadata, allow_nan=False)
    again, repeated = slice_landmarks(result, 1., 2.)
    assert again["time"].size == 0 and repeated["abstain"]


def test_no_usable_points_abstains_even_if_timestamps_exist():
    source = tracks()
    source["point_valid"][:] = False
    source["point_segment_id"][:] = -1
    result, metadata = slice_landmarks(source, 0, 1)
    assert result["time"].size == 11
    assert not metadata["empty"] and metadata["abstain"]
    assert metadata["abstain_reason"] == "no_valid_point_support"


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(time=np.zeros(11)),
    lambda value: value.update(time=np.r_[np.arange(10), np.nan]),
    lambda value: value.update(time=np.arange(11) - 1),
    lambda value: value.update(time=np.array([1, 0], dtype=np.uint64)),
    lambda value: value.update(points_m=np.zeros((11, 3))),
    lambda value: value.update(point_valid=np.ones(11, bool)),
    lambda value: value.update(point_observed=np.full((11, 11), np.nan)),
    lambda value: value.update(point_segment_id=np.full((11, 11), .5)),
    lambda value: value.update(landmark_signal_values=np.zeros((10, 11))),
    lambda value: value.update(landmark_signal_valid=np.ones((11, 10), bool)),
])
def test_bad_time_support_or_signal_shapes_fail(mutation):
    source = tracks()
    mutation(source)
    with pytest.raises(ValueError):
        slice_landmarks(source, 0, 1)
