"""Independent point-track tests; synthetic predictions, never an RGB model run."""

import copy
import json

import numpy as np
import pytest

from scripts_omnidance.exp.ms.motionscript_v2.landmarks import (
    CAMERA_FRAME,
    CANONICAL_FRAME,
    SPEC_SCHEMA,
    preprocess_landmarks,
    validate_point_spec,
)


def point_spec():
    return {"schema": SPEC_SCHEMA, "units": "m", "canonical_frame": CANONICAL_FRAME,
            "points": [{"name": "left_index_tip", "canonical": [.1, .2, .3]},
                       {"name": "right_thumb_base", "canonical": [-.1, .2, .3]}],
            "source": {"method": "synthetic_test_queries", "anatomy_verified": False}}


def raw(*, count=11, point_count=3, source_fps=10, timestamps=None):
    times = np.arange(count, dtype=np.float64) / source_fps if timestamps is None else np.asarray(timestamps)
    points = np.zeros((count, point_count, 3), dtype=np.float32)
    for point in range(point_count):
        points[:, point, 0] = times + point
        points[:, point, 1] = .5 + point * .01
        points[:, point, 2] = 3 + point * .1
    return {"points_m": points, "valid": np.ones((count, point_count), dtype=bool),
            "timestamps": times, "names": np.asarray([f"query_{index}" for index in range(point_count)]),
            "coordinate_frame": CAMERA_FRAME, "units": "m", "fps": source_fps}


def test_point_spec_has_explicit_frame_units_and_unverified_provenance():
    source = point_spec()
    normalized = validate_point_spec(source)
    assert normalized == source
    assert normalized is not source
    assert normalized["source"] is not source["source"]
    normalized["source"]["method"] = "changed"
    assert source["source"]["method"] == "synthetic_test_queries"
    json.dumps(normalized, allow_nan=False)


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(schema="unknown"),
    lambda value: value.update(units="mm"),
    lambda value: value.update(canonical_frame="world"),
    lambda value: value.update(points=[]),
    lambda value: value["points"][0].update(name="Left_Index_Tip"),
    lambda value: value["points"][0].update(name="left index tip"),
    lambda value: value["points"][1].update(name="left_index_tip"),
    lambda value: value["points"][0].update(canonical=[1, 2]),
    lambda value: value["points"][0].update(canonical=[1, 2, np.nan]),
    lambda value: value["points"][0].update(canonical=[True, 0, 0]),
    lambda value: value.update(source=["unverified"]),
    lambda value: value.update(source={"score": np.inf}),
])
def test_invalid_point_spec_is_rejected(mutation):
    source = point_spec()
    mutation(source)
    with pytest.raises(ValueError):
        validate_point_spec(source)


def test_arbitrary_68_points_retain_actual_predictions_without_fk_or_axis_flip():
    source = raw(count=5, point_count=68, source_fps=20)
    source["points_m"][:, 37] = [12.5, .75, 4.25]  # Not on any rest-body template.
    original = copy.deepcopy(source)
    arrays, metadata = preprocess_landmarks(source, smooth_seconds=0)
    assert arrays["points_m"].shape == (5, 68, 3)
    np.testing.assert_array_equal(arrays["points_m"], source["points_m"])
    np.testing.assert_array_equal(arrays["raw_points_m"], source["points_m"])
    np.testing.assert_array_equal(arrays["names"], source["names"])
    assert arrays["point_valid"].all() and arrays["point_observed"].all()
    assert not arrays["point_interpolated"].any()
    assert arrays["coordinate_frame"].item() == CAMERA_FRAME
    assert arrays["units"].item() == "m"
    assert arrays["points_m"].dtype == arrays["raw_points_m"].dtype == np.float32
    assert arrays["time"].dtype == arrays["raw_time"].dtype == np.float64
    assert arrays["point_segment_id"].dtype == np.int32
    assert metadata["coordinate_transform_applied"] is False
    assert metadata["world_coordinates_recovered"] is False
    assert metadata["point_names_are_verified_anatomy"] is False
    assert "no SMPL FK" in metadata["geometry_source"]
    np.testing.assert_array_equal(source["points_m"], original["points_m"])
    json.dumps(metadata, allow_nan=False)


def test_per_point_short_gap_interpolates_without_filling_other_tracks():
    source = raw(count=5, source_fps=20)
    source["valid"][2, 1] = False
    source["points_m"][2, 1] = np.nan
    source["valid"][:, 0] = False  # No root or parent-chain dependency.
    arrays, _ = preprocess_landmarks(source, max_gap_seconds=.05, smooth_seconds=0)
    assert not arrays["point_valid"][:, 0].any()
    assert np.isnan(arrays["points_m"][:, 0]).all()
    assert arrays["point_valid"][:, 1:].all()
    assert not arrays["point_observed"][2, 1]
    assert arrays["point_interpolated"][2, 1]
    assert arrays["point_observed"][2, 2]
    np.testing.assert_allclose(arrays["points_m"][2, 1], [1.1, .51, 3.1], atol=2e-6)
    np.testing.assert_array_equal(arrays["raw_valid"], source["valid"])
    assert np.isnan(arrays["raw_points_m"][2, 1]).all()


def test_long_gaps_and_edges_are_not_filled_or_smoothed_across():
    source = raw(count=21, source_fps=20)
    source["valid"][[0, 1, 7, 8, 9, 10, 11, 19, 20], 1] = False
    source["points_m"][12:, 1, 0] += 100
    arrays, metadata = preprocess_landmarks(source, max_gap_seconds=.1, smooth_seconds=.1)
    np.testing.assert_array_equal(arrays["point_valid"][:, 1], source["valid"][:, 1])
    assert np.isnan(arrays["points_m"][~arrays["point_valid"]]).all()
    assert np.all(arrays["point_segment_id"][~arrays["point_valid"]] == -1)
    assert arrays["points_m"][6, 1, 0] < 2
    assert arrays["points_m"][12, 1, 0] > 100
    assert metadata["per_point"][1]["long_gap_count"] == 1
    assert metadata["edge_extrapolation"] is False


def test_coarse_grid_preserves_independent_point_gap_boundaries():
    source = raw(count=6, source_fps=100)
    source["valid"][1:4, 1] = False
    arrays, _ = preprocess_landmarks(source, fps=20, max_gap_seconds=0, smooth_seconds=0)
    assert arrays["point_valid"].all()
    assert arrays["point_observed"].all()
    assert arrays["point_segment_id"][:, 1].tolist() == [0, 1]
    assert arrays["point_segment_id"][:, 0].tolist() == [0, 0]
    np.testing.assert_allclose(arrays["time"], [0, .05], rtol=0, atol=1e-15)


def test_resampling_uses_real_timestamps_and_exact_output_interval():
    source = raw(count=4, source_fps=10, timestamps=[.2, .3, .5, .56])
    arrays, _ = preprocess_landmarks(source, fps=20, max_gap_seconds=.2, smooth_seconds=0)
    np.testing.assert_allclose(arrays["time"], [.2, .25, .3, .35, .4, .45, .5, .55], atol=1e-15)
    np.testing.assert_allclose(np.diff(arrays["time"]), .05, rtol=0, atol=1e-15)
    assert arrays["time"][-1] <= source["timestamps"][-1]
    np.testing.assert_allclose(arrays["points_m"][:, 0, 0], arrays["time"], atol=2e-6)
    assert arrays["point_observed"][:, 0].tolist() == [True, False, True, False, False, False, True, False]
    np.testing.assert_array_equal(arrays["point_interpolated"], arrays["point_valid"] & ~arrays["point_observed"])


def test_smoothing_width_is_seconds_on_output_grid():
    first, _ = preprocess_landmarks(raw(count=17, source_fps=16), fps=20, smooth_seconds=.1)
    second, _ = preprocess_landmarks(raw(count=21, source_fps=20), fps=20, smooth_seconds=.1)
    np.testing.assert_allclose(first["time"], second["time"], atol=1e-15)
    np.testing.assert_allclose(first["points_m"], second["points_m"], atol=2e-6)


def test_uncertainty_is_preserved_but_gates_only_when_requested():
    source = raw()
    source["uncertainty"] = np.zeros((11, 3), dtype=np.float32)
    source["uncertainty"][:, 1] = 10
    ungated, _ = preprocess_landmarks(source, smooth_seconds=0)
    gated, metadata = preprocess_landmarks(source, smooth_seconds=0, max_point_uncertainty=1)
    assert ungated["point_valid"].all()
    assert not gated["point_valid"][:, 1].any()
    assert gated["point_valid"][:, [0, 2]].all()
    np.testing.assert_array_equal(gated["raw_uncertainty"], source["uncertainty"])
    assert gated["raw_valid"].all() and not gated["raw_quality_valid"][:, 1].any()
    assert metadata["uncertainty_is_calibrated_confidence"] is False
    assert metadata["uncertainty_resampled"] is False
    assert metadata["uncertainty_units"] == "model_native_unspecified"
    assert "uncertainty" not in gated  # No synthetic confidence after interpolation.
    del source["uncertainty"]
    with pytest.raises(ValueError, match="uncertainty is required"):
        preprocess_landmarks(source, max_point_uncertainty=1)


def test_nonfinite_prediction_is_invalid_even_if_upstream_mask_claims_valid():
    source = raw(count=5)
    source["points_m"][:, 1] = np.nan
    arrays, metadata = preprocess_landmarks(source)
    assert arrays["raw_valid"].all()
    assert not arrays["raw_quality_valid"][:, 1].any()
    assert not arrays["point_valid"][:, 1].any()
    assert np.isnan(arrays["points_m"][:, 1]).all()
    assert metadata["input_masked_nonfinite_count"] == 5


def test_static_single_point_and_all_invalid_tracks_are_well_formed():
    source = raw(count=1, point_count=1)
    arrays, metadata = preprocess_landmarks(source)
    assert arrays["point_valid"].tolist() == [[True]]
    assert arrays["point_observed"].tolist() == [[True]]
    assert arrays["point_segment_id"].tolist() == [[0]]
    source["valid"][:] = False
    missing, empty_metadata = preprocess_landmarks(source)
    assert not missing["point_valid"].any()
    assert np.isnan(missing["points_m"]).all()
    assert missing["point_segment_id"].tolist() == [[-1]]
    json.dumps(metadata, allow_nan=False)
    json.dumps(empty_metadata, allow_nan=False)


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(points_m=np.zeros((11, 3, 2))),
    lambda value: value.update(points_m=np.zeros((11, 0, 3))),
    lambda value: value.update(points_m=np.zeros((0, 3, 3))),
    lambda value: value.update(points_m=np.zeros((11, 3, 3), dtype=bool)),
    lambda value: value.update(valid=np.ones(11, dtype=bool)),
    lambda value: value.update(valid=np.full((11, 3), 2)),
    lambda value: value.update(valid=np.ones((11, 3), dtype=float)),
    lambda value: value.update(names=["point", "point", "other"]),
    lambda value: value.update(names=["point", "other"]),
    lambda value: value.update(names=["left hand", "other", "third"]),
    lambda value: value.update(timestamps=np.zeros(11)),
    lambda value: value.update(timestamps=np.r_[np.arange(10), np.inf]),
    lambda value: value.update(timestamps=np.arange(11) - 1),
    lambda value: value.update(units="mm"),
    lambda value: value.update(coordinate_frame="world"),
    lambda value: value.update(fps=0),
    lambda value: value.update(uncertainty=np.zeros((11, 2))),
])
def test_invalid_cache_shapes_names_units_frames_and_times_are_rejected(mutation):
    source = raw()
    mutation(source)
    with pytest.raises(ValueError):
        preprocess_landmarks(source)


@pytest.mark.parametrize("option,value", [("fps", 0), ("fps", True),
                                          ("max_gap_seconds", -1), ("smooth_seconds", np.inf),
                                          ("max_point_uncertainty", -1)])
def test_invalid_resampling_options_are_rejected(option, value):
    with pytest.raises(ValueError):
        preprocess_landmarks(raw(), **{option: value})
