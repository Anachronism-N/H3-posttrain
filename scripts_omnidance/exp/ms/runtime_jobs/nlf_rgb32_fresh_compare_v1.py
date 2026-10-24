"""CPU-only comparison of fresh-detector32 and the prior cached-box32 bundles."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import av
import cv2
import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from ms_landmarks import load_bundle, probe_video


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fresh', required=True)
    ap.add_argument('--cached', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    fresh_path, cached_path, out = map(Path, (args.fresh, args.cached, args.out))
    if out.exists():
        raise ValueError('Fresh independent comparison output required')
    f, fm, fsha = load_bundle(fresh_path)
    c, cm, csha = load_bundle(cached_path)
    if fm['source'] != 'nlf_fresh_detector_rgb_landmarks' or not fm['no_external_boxes_supplied']:
        raise ValueError('Fresh bundle must contain detector32, not a reused box prediction')
    if cm['source'] != 'nlf_direct_rgb_landmarks':
        raise ValueError('Comparison baseline must be prior cached-box32')
    for key in ('video', 'model', 'spec'):
        if fm['source_sha256'][key] != cm['source_sha256'][key] or sha(fm['source_paths'][key]) != fm['source_sha256'][key]:
            raise ValueError('Fresh and cached runs must have identical actual ' + key)
    if not np.array_equal(f['names'], c['names']) or not np.allclose(f['timestamps'], c['timestamps'], rtol=0, atol=1e-8):
        raise ValueError('Same32 query names and exact source PTS required')
    n, k = f['points_m'].shape[:2]
    if k != 32 or f['points_m'].shape != c['points_m'].shape or f['points2d_px'].shape != (n, 32, 2):
        raise ValueError('Direct32 geometry shapes differ')
    count, indices, offsets = f['detection_count'], f['selected_index'], f['detection_offsets']
    if len(offsets) != n + 1 or not np.array_equal(np.diff(offsets), count):
        raise ValueError('All-person ragged detection counts/offsets mismatch')
    for t in range(n):
        if count[t] == 0:
            if indices[t] != -1 or f['valid'][t].any() or not np.isnan(f['points_m'][t]).all():
                raise ValueError('No detection must retain missing coordinates/mask')
        else:
            boxes = f['all_boxes'][offsets[t]:offsets[t + 1]]
            chosen = int(np.argmax(boxes[:, 2] * boxes[:, 3]))
            if indices[t] != chosen or not np.array_equal(f['box'][t], boxes[chosen]):
                raise ValueError('Largest-area selection was not retained')
            q = f['all_points2d_px'][offsets[t] + chosen]
            if not np.array_equal(q, f['points2d_px'][t], equal_nan=True):
                raise ValueError('Selected output not bound to preserved people geometry')
    width, height = fm['frame_size']
    def necessary(a):
        p = a['points2d_px']
        return a['valid'] & np.isfinite(p).all(-1) & (p[:, :, 0] >= 0) & (p[:, :, 0] < width) & (p[:, :, 1] >= 0) & (p[:, :, 1] < height)
    fn, cn = necessary(f), necessary(c)
    out.mkdir(parents=True)
    np.savez_compressed(out / 'same_rgb_mask_comparison.npz', timestamps=f['timestamps'], names=f['names'],
        fresh_numeric_valid=f['valid'], fresh_in_frame_necessary=fn, cached_numeric_valid=c['valid'], cached_in_frame_necessary=cn,
        fresh_points_m=f['points_m'], cached_points_m=c['points_m'], fresh_points2d_px=f['points2d_px'], cached_points2d_px=c['points2d_px'],
        fresh_box=f['box'], cached_box=c['box'], detection_count=count, selected_index=indices)
    video = Path(fm['source_paths']['video'])
    info = probe_video(video, '/opt/ms_tools/ffprobe')
    frames, times = [], []
    with av.open(str(video)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format='bgr24'))
            times.append(float(frame.pts * frame.time_base))
    times = np.asarray(times)
    times -= times[0]
    if len(frames) != n or not np.allclose(times, f['timestamps'], rtol=0, atol=1e-8):
        raise ValueError('Decoded source PTS does not match predictions')
    pw, ph, hh, footer = 352, 640, 56, 56
    video_out = out / 'original_cached32_fresh_detector32.mp4'
    output_width, output_height = pw * 3, ph + hh + footer
    with (out / 'encoder.log').open('xb') as err:
        proc = subprocess.Popen(['/usr/local/bin/ffmpeg', '-nostdin', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{output_width}x{output_height}', '-r', str(info['fps']), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast',
            '-crf', '20', '-pix_fmt', 'yuv420p', str(video_out)], stdin=subprocess.PIPE, stderr=err)
        try:
            for t, source in enumerate(frames):
                canvas = np.zeros((output_height, output_width, 3), np.uint8)
                for panel, label in enumerate(('Original RGB', 'Prior cached boxes + direct32', 'Fresh detector + direct32')):
                    x0 = panel * pw
                    canvas[hh:hh + ph, x0:x0 + pw] = cv2.resize(source, (pw, ph))
                    cv2.putText(canvas, label, (x0 + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, .46, (255, 255, 255), 1)
                    cv2.putText(canvas, f'frame {t} PTS {times[t]:.4f}', (x0 + 5, 41), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
                for panel, a, mask in ((1, c, cn), (2, f, fn)):
                    for j in range(32):
                        if mask[t, j]:
                            pixel = a['points2d_px'][t, j]
                            x, y = int(pixel[0] * pw / width) + panel * pw, int(pixel[1] * ph / height) + hh
                            color = (40, 200, 70) if j < 22 else (255, 130, 50) if str(a['names'][j]).startswith('left') else (50, 100, 255)
                            cv2.circle(canvas, (x, y), 2 if j < 22 else 3, color, -1)
                            if j >= 22:
                                cv2.putText(canvas, str(j), (x + 3, y - 2), cv2.FONT_HERSHEY_SIMPLEX, .3, color, 1)
                cv2.putText(canvas, f'Fresh detections {count[t]} selected {indices[t]} | No tracking / no calibrated visibility / no GT',
                    (6, hh + ph + 18), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
                cv2.putText(canvas, f'Necessary in-frame support cached {cn[t].sum()}/32 fresh {fn[t].sum()}/32; occlusion/heel doubts unchanged',
                    (6, hh + ph + 39), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
                if t in (0, 32, n - 1):
                    if not cv2.imwrite(str(out / f'frame_{t:03d}.png'), canvas):
                        raise RuntimeError('Snapshot encoding failed')
                proc.stdin.write(canvas.tobytes())
        finally:
            proc.stdin.close()
            rc = proc.wait(timeout=60)
        if rc:
            raise RuntimeError('Video encoding failed')
    report = dict(schema='nlf-fresh-detector32-cachedbox32-compare-v1', source_video_path=str(video), source_video_sha256=fm['source_sha256']['video'],
        fresh_manifest_path=str(fresh_path / 'manifest.json'), fresh_manifest_sha256=fsha,
        cached_manifest_path=str(cached_path / 'manifest.json'), cached_manifest_sha256=csha,
        actual_fresh_detection=True, old_cache_conditioning_used=False, person_tracking=False,
        frame_count=n, fps=info['fps'], media_bounds=fm['media_bounds'],
        fresh_numeric_valid_pointframes=int(f['valid'].sum()), cached_numeric_valid_pointframes=int(c['valid'].sum()),
        fresh_necessary_pointframes=int(fn.sum()), cached_necessary_pointframes=int(cn.sum()),
        fresh_detection_count_histogram={str(x): int((count == x).sum()) for x in np.unique(count)},
        empty_detection_frames=np.flatnonzero(count == 0).tolist(), multiple_detection_frames=np.flatnonzero(count > 1).tolist(),
        fresh_outside_counts={str(name): int((f['valid'][:, j] & ~fn[:, j]).sum()) for j, name in enumerate(f['names'])},
        same_rgb_native2d_difference_quantiles_px=np.quantile(np.linalg.norm(f['points2d_px'] - c['points2d_px'], axis=2)[fn & cn], [.5, .9, .99]).tolist() if (fn & cn).any() else None,
        difference_is_accuracy_metric=False, in_frame_mask_is_visibility=False, occlusion_not_resolved=True,
        accuracy_approved=False, training_approved=False, source_or_raw_modified=False, gpu_used=False,
        artifacts={p.name: dict(path=str(p), sha256=sha(p)) for p in out.iterdir() if p.is_file() and p.suffix in ('.npz', '.mp4', '.png')},
        helper_sha256=sha(Path(__file__)))
    with (out / 'comparison_receipt.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write(chr(10))
    print(json.dumps(dict(out=str(out), actual_fresh_detection=True, detections=report['fresh_detection_count_histogram'],
        necessary_pointframes=report['fresh_necessary_pointframes'], accuracy_approved=False)), flush=True)


if __name__ == '__main__':
    main()
