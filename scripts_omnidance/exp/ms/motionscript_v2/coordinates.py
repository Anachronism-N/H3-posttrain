"""CPU preprocessing of cached NLF poses, with explicit observation masks.

The fixed camera-axis conversion is not a recovered camera-to-world transform.
Only the first 22 SMPL body rotations are observed; the SMPL-H finger rotations
remain at their template rest values. No legacy MotionScript code is imported.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp


_CAMERA_TO_Y_UP = np.diag([1.0, -1.0, -1.0])
_FOOT_IDS = np.array([7, 8, 10, 11])


def _scalar(value, name, *, positive=True):
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a finite scalar")
    value = float(array)
    if not np.isfinite(value) or (value <= 0 if positive else value < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return value


def _runs(mask, segment_id=None):
    mask = np.asarray(mask, dtype=bool)
    if segment_id is None:
        edges = np.diff(np.r_[False, mask, False].astype(np.int8))
        return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    # A coarse destination grid can skip every timestamp inside a long gap.
    # Adjacency on that grid must not reconnect distinct source support spans.
    same = np.asarray(segment_id)[1:] == np.asarray(segment_id)[:-1]
    starts = mask & ~np.r_[False, mask[:-1] & same]
    ends = mask & ~np.r_[mask[1:] & same, False]
    return list(zip(np.flatnonzero(starts), np.flatnonzero(ends) + 1))


def _segments(times, observed, maximum_interval):
    indices = np.flatnonzero(observed)
    if not len(indices):
        return []
    boundaries = np.flatnonzero(np.diff(times[indices]) > maximum_interval + 1e-10) + 1
    return np.split(indices, boundaries)


def _supported_grid(source_time, indices, time):
    if len(indices) == 1:
        return np.flatnonzero(np.isclose(time, source_time[indices[0]], rtol=0, atol=1e-9))
    return np.flatnonzero((time >= source_time[indices[0]] - 1e-10)
                         & (time <= source_time[indices[-1]] + 1e-10))


def _smooth_rotations(matrices, sigma_frames):
    if sigma_frames == 0 or len(matrices) < 2:
        return matrices
    quaternion = Rotation.from_matrix(matrices).as_quat()
    for index in range(1, len(quaternion)):
        if np.dot(quaternion[index - 1], quaternion[index]) < 0:
            quaternion[index] *= -1
    filtered = gaussian_filter1d(quaternion, sigma_frames, axis=0, mode="nearest", truncate=3)
    norm = np.linalg.norm(filtered, axis=1, keepdims=True)
    bad = norm[:, 0] < 1e-10
    filtered[bad] = quaternion[bad]
    norm[bad] = 1
    return Rotation.from_quat(filtered / norm).as_matrix()


def _interpolate(source_time, values, observed, time, maximum_interval, sigma_frames,
                 *, rotations=False):
    shape = (len(time),) + values.shape[1:]
    result = np.full(shape, np.nan, dtype=np.float64)
    usable = np.zeros(len(time), dtype=bool)
    for indices in _segments(source_time, observed, maximum_interval):
        grid = _supported_grid(source_time, indices, time)
        if not len(grid):
            continue
        query = np.clip(time[grid], source_time[indices[0]], source_time[indices[-1]])
        if rotations:
            if len(indices) == 1:
                selected = np.repeat(values[indices], len(grid), axis=0)
            else:
                selected = Slerp(source_time[indices], Rotation.from_matrix(values[indices]))(query).as_matrix()
            selected = _smooth_rotations(selected, sigma_frames)
        else:
            flattened = values[indices].reshape(len(indices), -1)
            selected = np.stack([np.interp(query, source_time[indices], column)
                                 for column in flattened.T], axis=1).reshape((len(grid),) + values.shape[1:])
            if sigma_frames > 0 and len(grid) > 1:
                selected = gaussian_filter1d(selected, sigma_frames, axis=0, mode="nearest", truncate=3)
        result[grid] = selected
        usable[grid] = True
    return result, usable


def _yaw_matrix(angle):
    angle = np.asarray(angle)
    result = np.zeros(angle.shape + (3, 3), dtype=np.float64)
    cosine, sine = np.cos(angle), np.sin(angle)
    result[..., 0, 0] = result[..., 2, 2] = cosine
    result[..., 0, 2], result[..., 2, 0] = sine, -sine
    result[..., 1, 1] = 1
    return result


def _forward_kinematics(rotations, joints, parents):
    count = len(rotations)
    global_rotation = np.zeros_like(rotations)
    positions = np.zeros((count, 52, 3), dtype=np.float64)
    global_rotation[:, 0] = rotations[:, 0]
    for joint in range(1, 52):
        parent = parents[joint]
        global_rotation[:, joint] = global_rotation[:, parent] @ rotations[:, joint]
        positions[:, joint] = positions[:, parent] + np.einsum(
            "tij,j->ti", global_rotation[:, parent], joints[joint] - joints[parent])
    return positions


def _floor_height(global_joints, joint_valid, valid, time, sigma_frames, segment_id):
    """Rolling lower foot envelope; deliberately not a calibrated floor plane."""
    feet = np.where(joint_valid[:, _FOOT_IDS], global_joints[:, _FOOT_IDS, 1], np.inf)
    foot_y = feet.min(axis=1)
    foot_y[~np.isfinite(foot_y)] = np.nan
    floor = np.full(len(time), np.nan)
    for start, end in _runs(valid, segment_id):
        for index in range(start, end):
            selection = (time[start:end] >= time[index] - 0.75) & (time[start:end] <= time[index] + 0.75)
            values = foot_y[start:end][selection]
            values = values[np.isfinite(values)]
            if len(values):
                floor[index] = np.percentile(values, 10)
        for left, right in _runs(np.isfinite(floor[start:end])):
            if sigma_frames > 0 and right - left > 1:
                selection = slice(start + left, start + right)
                floor[selection] = gaussian_filter1d(floor[selection], sigma_frames, mode="nearest", truncate=3)
    return floor


def preprocess(nlf: dict, template: dict, *, fps=20.0, max_gap_seconds=0.2,
               smooth_seconds=0.0625, max_joint_uncertainty=None) -> tuple[dict, dict]:
    """Convert NLF's metre-valued ``j3d`` and SMPL axis angles to masked arrays.

    ``valid`` concerns usable root/translation/heading and floor support;
    ``joint_valid`` additionally gates individual joints and their parent chain.
    Use ``joint_valid.all(axis=1)`` when a consumer needs a complete body.
    ``observed`` marks output timestamps matching reliable source root samples,
    even when smoothing was requested; it does not mean their values are raw.
    ``joint_observed`` requires the individual source joint and its entire
    parent chain to be usable at that exact timestamp; ``joint_interpolated``
    is the remaining usable joint support, including destination resampling.
    ``segment_id`` preserves root support spans even on a coarse output grid;
    invalid frames have ID -1. Consumers must not join events across IDs.
    ``joint_segment_id`` preserves the corresponding independent joint support
    spans; an otherwise valid root does not bridge a joint's long missing gap.

    A gap's duration is the interval between bracketing observations minus one
    nominal source-frame interval. Interpolation never extrapolates. After long
    gaps, heading branches restart relative to the reference: unobserved turns
    cannot be recovered. ``coords``' Euler channel has unavoidable Euler-chart
    singularities; ``root_rotation`` is the authoritative complete rotation.
    """
    fps = _scalar(fps, "fps")
    max_gap_seconds = _scalar(max_gap_seconds, "max_gap_seconds", positive=False)
    smooth_seconds = _scalar(smooth_seconds, "smooth_seconds", positive=False)
    if max_joint_uncertainty is not None:
        max_joint_uncertainty = _scalar(max_joint_uncertainty, "max_joint_uncertainty", positive=False)

    pose = np.asarray(nlf.get("pose"), dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 72 or len(pose) < 1:
        raise ValueError("pose must have shape [T,72] with T >= 1")
    source_count = len(pose)
    points = np.asarray(nlf.get("j3d"), dtype=np.float64)
    if points.ndim != 3 or points.shape[0] != source_count or points.shape[1] < 22 or points.shape[2] != 3:
        raise ValueError("j3d must have shape [T,J,3], J >= 22, in metres")
    supplied_valid = np.asarray(nlf.get("valid"))
    if supplied_valid.shape != (source_count,) or supplied_valid.dtype.kind not in "biu":
        raise ValueError("valid must be a boolean or binary integer [T] mask")
    if np.any((supplied_valid != 0) & (supplied_valid != 1)):
        raise ValueError("valid must contain only zero and one")
    supplied_valid = supplied_valid.astype(bool)

    if "fps" in nlf:
        source_fps = _scalar(nlf["fps"], "nlf fps")
    elif "timestamps" in nlf and source_count > 1:
        source_fps = None  # Infer only after checking the timestamp array.
    else:
        raise ValueError("nlf fps is required unless multiple timestamps are supplied")
    if "timestamps" in nlf:
        source_time = np.asarray(nlf["timestamps"], dtype=np.float64)
        if (source_time.shape != (source_count,) or not np.isfinite(source_time).all()
                or np.any(np.diff(source_time) <= 0) or source_time[0] < 0):
            raise ValueError("timestamps must be finite, nonnegative and strictly increasing [T] seconds")
        if source_fps is None:
            source_fps = 1.0 / float(np.median(np.diff(source_time)))
    else:
        source_time = np.arange(source_count, dtype=np.float64) / source_fps

    joints = np.asarray(template.get("J_template"), dtype=np.float64)
    parents = np.asarray(template.get("parents"))
    if joints.shape != (52, 3) or not np.isfinite(joints).all():
        raise ValueError("J_template must be finite [52,3] metre-valued SMPL-H rest joints")
    if (parents.shape != (52,) or parents.dtype.kind not in "iu" or parents[0] not in (-1, 0)
            or any(not 0 <= parents[j] < j for j in range(1, 52))):
        raise ValueError("parents must be a topologically ordered [52] SMPL-H tree")
    parents = parents.astype(np.int64)

    # The endpoint is never moved to make the output appear to have a target fps.
    output_count = int(np.floor((source_time[-1] - source_time[0]) * fps + 1e-10)) + 1
    time = source_time[0] + np.arange(output_count, dtype=np.float64) / fps
    time = time[time <= source_time[-1]]
    maximum_interval = 1.0 / source_fps + max_gap_seconds
    sigma_frames = smooth_seconds * fps
    source_joint_valid = supplied_valid[:, None] & np.isfinite(pose[:, :66].reshape(-1, 22, 3)).all(axis=2)
    source_joint_valid &= np.isfinite(points[:, :22]).all(axis=2)
    uncertainty_gated = max_joint_uncertainty is not None
    uncertainty = None
    if "unc" in nlf:
        uncertainty = np.asarray(nlf["unc"], dtype=np.float64)
        if uncertainty.shape != points.shape[:2]:
            raise ValueError("unc must have shape [T,J] matching j3d")
    if uncertainty_gated:
        if uncertainty is None:
            raise ValueError("unc is required when max_joint_uncertainty is supplied")
        source_joint_valid &= np.isfinite(uncertainty[:, :22]) & (uncertainty[:, :22] >= 0)
        source_joint_valid &= uncertainty[:, :22] <= max_joint_uncertainty
    source_valid = source_joint_valid[:, 0].copy()
    source_joint_valid &= source_valid[:, None]
    for joint in range(1, 22):
        source_joint_valid[:, joint] &= source_joint_valid[:, parents[joint]]
    source_segments = _segments(source_time, source_valid, maximum_interval)
    segment_id = np.full(len(time), -1, dtype=np.int32)
    for identifier, indices in enumerate(source_segments):
        segment_id[_supported_grid(source_time, indices, time)] = identifier
    boxes = None
    if "box" in nlf:
        boxes = np.asarray(nlf["box"], dtype=np.float64)
        if boxes.ndim != 2 or boxes.shape[0] != source_count or boxes.shape[1] not in (4, 5):
            raise ValueError("box must have shape [T,4] or [T,5]")

    source_rotations = np.full((source_count, 22, 3, 3), np.nan)
    local_angles = pose[:, :66].reshape(-1, 22, 3)
    for joint in range(22):
        usable = source_joint_valid[:, joint]
        if usable.any():
            source_rotations[usable, joint] = Rotation.from_rotvec(local_angles[usable, joint]).as_matrix()
    source_rotations[:, 0] = _CAMERA_TO_Y_UP @ source_rotations[:, 0]
    pelvis_source = points[:, 0] @ _CAMERA_TO_Y_UP.T
    pelvis, valid = _interpolate(source_time, pelvis_source, source_valid, time,
                                 maximum_interval, sigma_frames)
    rotations = np.broadcast_to(np.eye(3), (len(time), 52, 3, 3)).copy()
    joint_valid = np.zeros((len(time), 22), dtype=bool)
    joint_segment_id = np.full((len(time), 22), -1, dtype=np.int32)
    for joint in range(22):
        values, usable = _interpolate(source_time, source_rotations[:, joint], source_joint_valid[:, joint],
                                     time, maximum_interval, sigma_frames, rotations=True)
        rotations[usable, joint] = values[usable]
        joint_valid[:, joint] = usable
        for identifier, indices in enumerate(_segments(source_time, source_joint_valid[:, joint], maximum_interval)):
            joint_segment_id[_supported_grid(source_time, indices, time), joint] = identifier
    valid &= joint_valid[:, 0]
    root_rotation = rotations[:, 0].copy()
    horizontal_forward = np.linalg.norm(root_rotation[:, [0, 2], 2], axis=1)
    valid &= horizontal_forward > 1e-6  # Heading is undefined when the root forward axis is vertical.
    joint_valid[:, 0] &= valid
    for joint in range(1, 22):
        joint_valid[:, joint] &= joint_valid[:, parents[joint]]
    positions = _forward_kinematics(rotations, joints, parents)
    global_joints = positions[:, :22] + pelvis[:, None]
    floor = _floor_height(global_joints, joint_valid, valid, time, sigma_frames, segment_id)
    valid &= np.isfinite(floor)
    joint_valid &= valid[:, None]

    yaw_deg = np.full(len(time), np.nan)
    translation = np.full((len(time), 3), np.nan)
    body52 = np.full((len(time), 52, 3), np.nan)
    euler = np.full((len(time), 3), np.nan)
    reference_index = int(np.flatnonzero(valid)[0]) if valid.any() else None
    if reference_index is not None:
        heading = np.arctan2(root_rotation[:, 0, 2], root_rotation[:, 2, 2])
        reference_heading = heading[reference_index]
        reference_rotation = _yaw_matrix(-reference_heading)
        translation[valid] = (pelvis[valid] - pelvis[reference_index]) @ reference_rotation.T
        to_z_up = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        for start, end in _runs(valid, segment_id):
            relative = np.arctan2(np.sin(heading[start:end] - reference_heading),
                                  np.cos(heading[start:end] - reference_heading))
            yaw_deg[start:end] = np.degrees(np.unwrap(relative))
            body52[start:end] = np.einsum("tij,tkj->tki", _yaw_matrix(-heading[start:end]), positions[start:end])
            body52[start:end, :, 1] += (pelvis[start:end, 1] - floor[start:end])[:, None]
            relative_root = reference_rotation @ root_rotation[start:end]
            as_z_up = to_z_up @ relative_root @ to_z_up.T
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)  # Euler-chart limitation is recorded below.
                angles = Rotation.from_matrix(as_z_up).as_euler("xyz")
            euler[start:end] = np.degrees(np.unwrap(angles, axis=0))

    full_joint_valid = np.ones((len(time), 52), dtype=bool)
    full_joint_valid[:, :22] = joint_valid
    for joint in range(22, 52):
        full_joint_valid[:, joint] = full_joint_valid[:, parents[joint]]
    body52[~full_joint_valid] = np.nan
    global_joints[~joint_valid] = np.nan
    root_rotation[~valid] = np.nan
    coords = np.concatenate([body52, euler[:, None], translation[:, None]], axis=1)

    observed = np.zeros(len(time), dtype=bool)
    for source_index in np.flatnonzero(source_valid):
        observed |= np.isclose(time, source_time[source_index], rtol=0, atol=1e-9)
    observed &= valid
    joint_observed = np.zeros_like(joint_valid)
    for source_index in np.flatnonzero(source_valid):
        matches = np.isclose(time, source_time[source_index], rtol=0, atol=1e-9)
        joint_observed[matches] |= source_joint_valid[source_index]
    joint_observed &= joint_valid
    segment_id[~valid] = -1
    joint_segment_id[~joint_valid] = -1
    gap_runs = _runs(~source_valid)
    arrays = dict(time=time, body=body52[:, :22].astype(np.float32),
                  translation=translation.astype(np.float32), yaw_deg=yaw_deg.astype(np.float32),
                  valid=valid, joint_valid=joint_valid, observed=observed,
                  joint_observed=joint_observed, joint_interpolated=joint_valid & ~joint_observed,
                  interpolated=valid & ~observed, source_time=source_time.copy(),
                  source_valid=source_valid, source_detection_valid=supplied_valid.copy(),
                  source_joint_valid=source_joint_valid, segment_id=segment_id,
                  joint_segment_id=joint_segment_id,
                  root_rotation=root_rotation.astype(np.float32),
                  global_joints=global_joints.astype(np.float32), coords=coords.astype(np.float32))
    if uncertainty is not None:
        arrays["source_uncertainty"] = uncertainty.astype(np.float32)
    if boxes is not None:
        arrays["source_boxes"] = boxes.astype(np.float32)
    metadata = dict(
        format_version=1, source_fps=float(source_fps), output_fps=fps,
        source_timestamps="provided_seconds" if "timestamps" in nlf else "frame_index_over_fps",
        source_frames=source_count, output_frames=len(time), source_valid_frames=int(source_valid.sum()),
        valid_frames=int(valid.sum()), observed_frames=int(observed.sum()),
        interpolated_frames=int((valid & ~observed).sum()),
        source_valid_ratio=float(source_valid.mean()), valid_ratio=float(valid.mean()),
        source_missing_runs=len(gap_runs), supported_source_segments=len(source_segments),
        long_gap_count=max(0, len(source_segments) - 1), output_valid_segments=len(_runs(valid, segment_id)),
        output_geometry_dtype="float32", output_time_dtype="float64",
        segment_id_definition="source root support segment; invalid=-1; adjacent output frames can have distinct IDs",
        joint_segment_id_definition="per-joint source support segment after parent-chain quality gating; joint-invalid=-1; IDs local to each joint",
        max_gap_seconds=max_gap_seconds,
        gap_duration_definition="bracketing_observation_interval_minus_one_nominal_source_frame",
        edge_extrapolation=False, smooth_seconds=smooth_seconds,
        smoothing="output-grid Gaussian positions; sign-continuous normalized Gaussian quaternions",
        smoothing_crosses_unsupported_gap=False,
        coordinate_system="Y-up metres after diag(1,-1,-1) camera-axis rotation; not recovered world coordinates",
        body_frame="root-relative, per-frame projected-forward yaw removed, estimated pelvis height above floor added",
        global_joints_frame="fixed Y-up camera axes, absolute pelvis translation retained; not world ground truth",
        root_rotation_frame="complete root rotation in fixed Y-up camera axes",
        translation_frame="pelvis displacement from first usable output sample, rotated by its negative heading",
        reference_time=None if reference_index is None else float(time[reference_index]),
        yaw_frame="relative to first usable output heading; unwrapped separately within each valid segment",
        unobserved_turns_across_long_gaps="unknown; each segment restarts on the reference principal branch",
        coords_layout="52 SMPL-H template joints, root Euler XYZ in Z-up first-heading frame (degrees), translation",
        euler_limitation="unwrapped per segment; Euler-chart/gimbal singularities remain; use root_rotation for geometry",
        floor_method="1.5-second rolling 10th percentile of trusted foot minima, within valid segments only",
        floor_limitations="heuristic; no calibrated gravity/camera/floor plane; prolonged airborne motion, camera motion and terrain can bias height",
        template_shape="fixed J_template; NLF betas not applied",
        hand_fingers_observed=False, finger_rotations="unobserved template-rest local rotations",
        uncertainty_gate_enabled=uncertainty_gated, max_joint_uncertainty=max_joint_uncertainty,
        uncertainty_is_calibrated_confidence=False,
        valid_definition="usable root/pelvis/heading and floor support; individual joints additionally require joint_valid",
        joint_valid_definition="finite source pose/j3d, optional uncertainty gate, bounded interpolation, and parent-chain support",
        observed_definition="output timestamp matches a reliable source root sample; optional smoothing can change its values",
        joint_observed_definition="exact source timestamp with usable joint and parent chain; optional smoothing may change values",
        joint_interpolated_definition="joint_valid and not joint_observed, including resampling and bounded gap interpolation",
        source_joint_valid_definition="source detection plus finite pose/j3d, optional uncertainty gate, and parent-chain support",
        source_uncertainty_preserved=uncertainty is not None, source_boxes_preserved=boxes is not None,
        original_source_valid_frames=int(supplied_valid.sum()),
    )
    return arrays, metadata
