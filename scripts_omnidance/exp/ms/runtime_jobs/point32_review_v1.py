import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys

import av
import cv2
import numpy as np

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
sys.path.insert(0, str(ROOT / 'H3-posttrain/scripts_omnidance/exp/ms'))
from ms_landmarks import load_bundle, probe_video
from motionscript_v2.landmark_events import measure_landmark_signals


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def project(points, width, height):
    focal = max(width, height) / (2 * np.tan(np.deg2rad(55.) / 2))
    good = np.isfinite(points).all(-1) & (points[..., 2] > 1e-6)
    safe_z = np.where(good, points[..., 2], 1)
    pixels = points[..., :2] / safe_z[..., None] * focal + [(width - 1) / 2, (height - 1) / 2]
    return pixels, good


def frame_indices(time, stop, fps):
    output_time = np.arange(math.ceil(stop * fps), dtype=float) / fps
    indices = np.searchsorted(time, output_time, side='right') - 1
    return output_time, np.clip(indices, 0, len(time) - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--legacy-nlf', required=True)
    parser.add_argument('--parent-receipt', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--ffprobe', default='/opt/ms_tools/ffprobe')
    parser.add_argument('--ffmpeg', default='/usr/local/bin/ffmpeg')
    parser.add_argument('--output-fps', type=float, default=30.)
    args = parser.parse_args()
    bundle, old_path, out = map(Path, (args.bundle, args.legacy_nlf, args.out))
    if out.exists():
        raise ValueError('New independent review output required')
    arrays, manifest, manifest_sha = load_bundle(bundle)
    if manifest['source'] != 'nlf_direct_rgb_landmarks' or manifest['point_count'] != 32:
        raise ValueError('Must consume genuine direct32 bundle')
    if manifest['smpl_fitting_used'] or manifest['template_fk_used']:
        raise ValueError('Direct predictions must not use fitting or template FK')
    video = Path(manifest['source_paths']['video'])
    if old_path.resolve() != Path(manifest['source_paths']['boxes']).resolve():
        raise ValueError('Legacy comparison must use the identical extraction box/pose cache')
    for label, path in manifest['source_paths'].items():
        if sha(path) != manifest['source_sha256'][label]:
            raise ValueError('Changed source: ' + label)
    old_bytes = old_path.read_bytes()
    with np.load(io.BytesIO(old_bytes), allow_pickle=False) as cache:
        old = dict(cache)
    info = probe_video(video, args.ffprobe)
    time = arrays['timestamps']
    points, pixels, valid = arrays['points_m'], arrays['points2d_px'], arrays['valid']
    names = arrays['names'].tolist()
    n, k = len(time), len(names)
    if (n != 81 or k != 32 or points.shape != (n, k, 3) or pixels.shape != (n, k, 2)
            or valid.shape != (n, k) or arrays['uncertainty'].shape != (n, k)
            or old['pose'].shape != (n, 72) or old['j3d'].shape != (n, 24, 3)
            or old['valid'].shape != (n,) or float(old['fps']) != 16
            or not np.allclose(time, info['time'], atol=1e-7, rtol=0)):
        raise ValueError('Actual direct/legacy/media shapes or time span mismatch')
    if not valid.any() or not np.isfinite(points[valid]).all() or not np.isfinite(pixels[valid]).all():
        raise ValueError('Nonfinite valid direct prediction')
    bounds = manifest['media_bounds']
    if bounds['end'] is None or abs(bounds['end'] - 5.0625) > 1e-7:
        raise ValueError('Expected actual fixed3 media end 5.0625 seconds')
    parent_receipt = json.loads(Path(args.parent_receipt).read_text())
    if parent_receipt['model_sha256'] != manifest['source_sha256']['model']:
        raise ValueError('Legacy parent chain belongs to a different model')
    parents = np.asarray(parent_receipt['parents'], np.int64)
    if parents.shape != (24,) or any(not 0 <= parents[j] < j for j in range(1, 24)):
        raise ValueError('Invalid model-attested legacy SMPL parents')
    local = np.full((n, 24, 3, 3), np.nan)
    global_rotation = local.copy()
    for t in np.flatnonzero(old['valid']):
        if not np.isfinite(old['pose'][t]).all():
            raise ValueError('Invalid legacy pose marked valid')
        for j, rotvec in enumerate(old['pose'][t].reshape(24, 3)):
            local[t, j] = cv2.Rodrigues(rotvec.astype(float))[0]
            global_rotation[t, j] = local[t, j] if j == 0 else global_rotation[t, parents[j]] @ local[t, j]
    segment = np.full_like(valid, -1, dtype=np.int64)
    for j in range(k):
        run = -1
        previous = False
        for t in range(n):
            if valid[t, j] and not previous:
                run += 1
            if valid[t, j]:
                segment[t, j] = run
            previous = bool(valid[t, j])
    measured, trace_meta = measure_landmark_signals(dict(
        time=time, points_m=points, names=arrays['names'], point_valid=valid,
        point_segment_id=segment, point_observed=valid,
        point_interpolated=np.zeros_like(valid), coordinate_frame=arrays['coordinate_frame'], units=arrays['units']))
    source_frames, decoded_time = [], []
    with av.open(str(video)) as container:
        for frame in container.decode(video=0):
            source_frames.append(frame.to_ndarray(format='bgr24'))
            decoded_time.append(float(frame.pts * frame.time_base))
    decoded_time = np.asarray(decoded_time)
    decoded_time -= decoded_time[0]
    if len(source_frames) != n or not np.allclose(decoded_time, time, atol=1e-7, rtol=0):
        raise ValueError('CPU decoded video PTS differs from direct raw PTS')
    width, height = info['width'], info['height']
    legacy_pixels, legacy_good = project(old['j3d'], width, height)
    legacy_good &= old['valid'][:, None]
    inbounds = ((pixels[..., 0] >= 0) & (pixels[..., 0] < width)
                & (pixels[..., 1] >= 0) & (pixels[..., 1] < height))
    projected_direct, projected_good = project(points, width, height)
    projection_residual = np.linalg.norm(projected_direct - pixels, axis=-1)
    residual_good = projected_good & valid
    out.mkdir(parents=True)
    np.savez_compressed(out / 'same_window_trace.npz', timestamps=time,
        direct_points_m=points, direct_points2d_px=pixels, direct_valid=valid,
        direct_uncertainty=arrays['uncertainty'], names=arrays['names'],
        legacy_pose72=old['pose'], legacy_j3d24_m=old['j3d'], legacy_valid=old['valid'],
        legacy_local_rotations24=local, legacy_global_rotations24=global_rotation,
        legacy_projected2d_default55=legacy_pixels, legacy_parent24=parents, **measured)
    dump(out / 'trace_metadata.json', trace_meta)
    pw, ph, header, footer = 352, 640, 72, 164
    output_width, output_height = pw * 3, ph + header + footer
    output_time, source_indices = frame_indices(time, bounds['end'], args.output_fps)
    video_out = out / 'rgb_direct32_vs_legacy_pose.mp4'
    error_path = out / 'ffmpeg.log'
    titles = ('Original RGB', 'Direct RGB body22+palm4+foot6', 'Legacy j3d22 + full pose24 axes')
    parent_edges = [(int(parents[j]), j) for j in range(1, 22)]
    lookup = {name: i for i, name in enumerate(names)}
    extra_edges = []
    for side in ('left', 'right'):
        wrist, index, little = [lookup[side + '_' + name] for name in ('wrist', 'index_mcp', 'little_mcp')]
        heel, big, small = [lookup[side + '_' + name] for name in ('heel', 'big_toe', 'little_toe')]
        extra_edges.extend([(wrist, index), (wrist, little), (index, little), (heel, big), (heel, small), (big, small)])
    signal_names = measured['landmark_signal_names'].tolist()
    signal_values = measured['landmark_signal_values']
    with error_path.open('xb') as errors:
        process = subprocess.Popen([args.ffmpeg, '-nostdin', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', str(output_width) + 'x' + str(output_height), '-r', str(args.output_fps), '-i', 'pipe:0',
            '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p', str(video_out)],
            stdin=subprocess.PIPE, stderr=errors)
        try:
            for output_t, source_index in zip(output_time, source_indices):
                t = int(source_index)
                canvas = np.zeros((output_height, output_width, 3), np.uint8)
                original = cv2.resize(source_frames[t], (pw, ph))
                for panel in range(3):
                    x0 = panel * pw
                    canvas[header:header + ph, x0:x0 + pw] = original
                    cv2.putText(canvas, titles[panel], (x0 + 5, 22), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
                    cv2.putText(canvas, 'PTS %.4f  frame %d/%d' % (time[t], t, n - 1), (x0 + 5, 43), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
                def xy(pixel, panel):
                    return (int(round(pixel[0] * pw / width)) + panel * pw, int(round(pixel[1] * ph / height)) + header)
                def ok(pixel):
                    return np.isfinite(pixel).all() and 0 <= pixel[0] < width and 0 <= pixel[1] < height
                for a, b in parent_edges + extra_edges:
                    if valid[t, a] and valid[t, b] and ok(pixels[t, a]) and ok(pixels[t, b]):
                        color = (0, 255, 255) if a >= 22 or b >= 22 else (50, 190, 50)
                        cv2.line(canvas, xy(pixels[t, a], 1), xy(pixels[t, b], 1), color, 1, cv2.LINE_AA)
                for j in range(k):
                    if valid[t, j] and ok(pixels[t, j]):
                        color = (50, 190, 50) if j < 22 else (255, 120, 50) if names[j].startswith('left') else (40, 100, 255)
                        pt = xy(pixels[t, j], 1)
                        cv2.circle(canvas, pt, 2 if j < 22 else 3, color, -1)
                        if j >= 22:
                            cv2.putText(canvas, str(j), (pt[0] + 3, pt[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, .3, color, 1)
                for a, b in parent_edges:
                    if legacy_good[t, a] and legacy_good[t, b] and ok(legacy_pixels[t, a]) and ok(legacy_pixels[t, b]):
                        cv2.line(canvas, xy(legacy_pixels[t, a], 2), xy(legacy_pixels[t, b], 2), (170, 220, 60), 1)
                for j in (7, 8, 10, 11, 20, 21, 22, 23):
                    if not old['valid'][t] or not np.isfinite(global_rotation[t, j]).all():
                        continue
                    origin = old['j3d'][t, j]
                    ends = origin[None] + .07 * global_rotation[t, j].T
                    end_px, end_ok = project(ends, width, height)
                    if legacy_good[t, j] and ok(legacy_pixels[t, j]):
                        for axis, color in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0))):
                            if end_ok[axis] and ok(end_px[axis]):
                                cv2.arrowedLine(canvas, xy(legacy_pixels[t, j], 2), xy(end_px[axis], 2), color, 1, tipLength=.25)
                cv2.putText(canvas, 'Uncalibrated geometry; occlusion/visibility UNKNOWN; no accuracy or contact approval',
                    (6, header + ph + 18), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
                cv2.putText(canvas, 'Direct valid %d/32  out-of-frame valid %d | Legacy projection: official default 55deg FOV, NOT GT' %
                    (int(valid[t].sum()), int((valid[t] & ~inbounds[t]).sum())),
                    (6, header + ph + 38), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
                for line, side in enumerate(('left', 'right')):
                    trace_text = []
                    for name in (side + '_palm.palm_normal_to_forearm_angle', side + '_foot.sole_normal_to_shin_angle', side + '_foot.foot_long_axis_to_shin_angle'):
                        if name in signal_names:
                            value = signal_values[t, signal_names.index(name)]
                            trace_text.append(name.split('.')[1] + '=' + ('%.1f' % value if np.isfinite(value) else 'unknown'))
                    cv2.putText(canvas, side + ': ' + ' '.join(trace_text), (6, header + ph + 59 + line * 19), cv2.FONT_HERSHEY_SIMPLEX, .36, (220, 220, 220), 1)
                for line, indices in enumerate((range(22, 26), range(26, 32))):
                    legend = ' '.join(str(j) + ':' + names[j] for j in indices)
                    cv2.putText(canvas, legend, (6, header + ph + 104 + line * 18), cv2.FONT_HERSHEY_SIMPLEX, .37, (200, 200, 200), 1)
                if t in (0, 32, 80) and (out / ('frame_%03d.png' % t)).exists() is False:
                    if not cv2.imwrite(str(out / ('frame_%03d.png' % t)), canvas):
                        raise RuntimeError('Failed to write local review snapshot')
                if t == 32 and not (out / 'hand_foot_roi_frame032.png').exists():
                    roi_sheet = np.zeros((304, 1024, 3), np.uint8)
                    groups = [('left palm', ('left_wrist', 'left_index_mcp', 'left_little_mcp')),
                              ('right palm', ('right_wrist', 'right_index_mcp', 'right_little_mcp')),
                              ('left foot', ('left_heel', 'left_big_toe', 'left_little_toe')),
                              ('right foot', ('right_heel', 'right_big_toe', 'right_little_toe'))]
                    for column, (label, point_names) in enumerate(groups):
                        xx = column * 256
                        cv2.putText(roi_sheet, label + ' predicted ROI', (xx + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
                        js = [lookup[name] for name in point_names]
                        if valid[t, js].all() and inbounds[t, js].all():
                            center = pixels[t, js].mean(axis=0)
                            half = 64 if 'foot' in label else 48
                            x0, y0 = max(0, int(center[0]) - half), max(0, int(center[1]) - half)
                            x1, y1 = min(width, int(center[0]) + half), min(height, int(center[1]) + half)
                            patch = cv2.resize(source_frames[t][y0:y1, x0:x1], (256, 256))
                            for j in js:
                                point = (int((pixels[t, j, 0] - x0) * 256 / (x1 - x0)), int((pixels[t, j, 1] - y0) * 256 / (y1 - y0)))
                                color = (255, 120, 50) if label.startswith('left') else (40, 100, 255)
                                cv2.circle(patch, point, 4, color, -1)
                                cv2.putText(patch, str(j), (point[0] + 3, point[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
                            roi_sheet[24:280, xx:xx + 256] = patch
                        else:
                            cv2.putText(roi_sheet, 'unknown / invalid ROI', (xx + 4, 100), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
                    cv2.putText(roi_sheet, 'Frame32 PTS2.0s; crop from raw RGB around predicted landmarks; NOT visibility/GT', (4, 298), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
                    if not cv2.imwrite(str(out / 'hand_foot_roi_frame032.png'), roi_sheet):
                        raise RuntimeError('Failed to write local ROI snapshot')
                process.stdin.write(canvas.tobytes())
        finally:
            process.stdin.close()
            code = process.wait(timeout=60)
        if code:
            raise RuntimeError('ffmpeg failed; see ' + str(error_path))
    perpoint = []
    for j, name in enumerate(names):
        good = valid[:, j]
        u = arrays['uncertainty'][:, j][good]
        perpoint.append(dict(index=j, name=name, finite_valid_frames=int(good.sum()),
            valid_out_of_bounds_frames=int((good & ~inbounds[:, j]).sum()),
            uncertainty_quantiles=np.quantile(u, [.1, .5, .9]).tolist() if len(u) else None,
            max_frame_jump_m=float(np.max(np.linalg.norm(np.diff(points[:, j], axis=0)[good[1:] & good[:-1]], axis=1))) if (good[1:] & good[:-1]).any() else None))
    plane_diagnostics = []
    for side in ('left', 'right'):
        for kind, point_names in [('palm', ('wrist', 'index_mcp', 'little_mcp')), ('foot', ('heel', 'big_toe', 'little_toe'))]:
            js = [lookup[side + '_' + name] for name in point_names]
            a, b, c = [points[:, j] for j in js]
            u, v = b - a, c - a
            lengths = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
            cross_norm = np.linalg.norm(np.cross(u, v), axis=1)
            good = valid[:, js].all(axis=1) & np.isfinite(cross_norm)
            sine = cross_norm / np.maximum(lengths, 1e-12)
            degenerate = good & ((lengths < 1e-10) | (sine < .10))
            plane_diagnostics.append(dict(side=side, kind=kind, support_frames=int(good.sum()),
                raw_cross_norm_m2_quantiles=np.quantile(cross_norm[good], [.1, .5, .9]).tolist() if good.any() else None,
                raw_plane_sine_quantiles=np.quantile(sine[good], [.1, .5, .9]).tolist() if good.any() else None,
                degenerate_frames=np.flatnonzero(degenerate).tolist(), raw_point_indices=js,
                flags_are_calibrated_visibility=False))
    frame_norm = np.linalg.norm(points, axis=2)
    true_points = points[valid]
    raw_statistics = dict(point_norm_m_quantiles=np.quantile(frame_norm[valid], [.01, .5, .99]).tolist(),
        xyz_min_m=true_points.min(axis=0).tolist(), xyz_max_m=true_points.max(axis=0).tolist(),
        frame_valid_points_min=int(valid.sum(axis=1).min()), frame_valid_points_max=int(valid.sum(axis=1).max()),
        all32_valid_frame_indices=np.flatnonzero(valid.all(axis=1)).tolist(),
        any_invalid_frame_indices=np.flatnonzero(~valid.all(axis=1)).tolist(),
        legacy_valid_frames=int(old['valid'].sum()), direct_plane_diagnostics=plane_diagnostics)
    inputs = {label: dict(path=str(path), sha256=sha(path)) for label, path in
        [('bundle_manifest', bundle / 'manifest.json'), ('raw', bundle / 'raw.npz'), ('legacy_nlf', old_path), ('parent_receipt', Path(args.parent_receipt)), ('video', video), ('review_script', Path(__file__))]}
    receipt = dict(schema='nlf-direct32-same-window-review-v1', actual_direct32_loaded=True,
        bundle_manifest_sha256=manifest_sha, inputs=inputs, source_frames=n,
        source_fps=float(arrays['fps']), media_bounds=bounds, names=names, point_stats=perpoint,
        raw_statistics=raw_statistics, display_only_resize=[352, 640], source_decode_resize_or_rotation=False,
        legacy_full_pose72_preserved=True, legacy_global_rotation24_saved=True,
        smpl_fk_used_for_new_points=False, camera_projection='NLF pinned ptu3d intrinsic_matrix_from_field_of_view: focal=max(H,W)/(2*tan(55deg/2)), center=((W-1)/2,(H-1)/2); default assumption not camera calibration',
        direct_projection_internal_consistency_px_quantiles=np.quantile(projection_residual[residual_good], [.5, .9, .99]).tolist(),
        projection_residual_is_accuracy_metric=False, output_fps=args.output_fps,
        output_frames=len(output_time), output_duration_rounding_seconds=len(output_time) / args.output_fps - bounds['end'],
        output_temporal_method='Zero-order hold source frame selected by actual PTS, no motion interpolation',
        source_frames_filled_or_interpolated=False, audio_muxed=False,
        artifacts={name: dict(path=str(out / name), sha256=sha(out / name)) for name in ('same_window_trace.npz', 'trace_metadata.json', 'rgb_direct32_vs_legacy_pose.mp4', 'frame_000.png', 'frame_032.png', 'frame_080.png', 'hand_foot_roi_frame032.png')},
        visibility='unknown', semantic_approval=False, accuracy_review_completed=False, gpu_used=False)
    dump(out / 'review_receipt.json', receipt)
    print(json.dumps(dict(out=str(out), raw_verified=True, source_frames=n, output_frames=len(output_time),
        mean_valid_points=float(valid.sum(axis=1).mean()), overlay_path=str(video_out), semantic_approval=False)))


if __name__ == '__main__':
    main()
