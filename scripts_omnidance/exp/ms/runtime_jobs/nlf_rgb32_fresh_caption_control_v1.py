"""Isolated numeric/in-frame support and caption controls; no new inference."""
import collections
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from ms_landmarks import load_bundle
from motionscript_v2.landmarks import preprocess_landmarks
from motionscript_v2.landmark_events import extract_landmark_events, measure_landmark_signals, render_landmark_caption
from motionscript_v2.chunks import slice_landmarks

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS')
FRESH = ROOT / 'nlf_rgb32_fresh_detector_fixed3_v1'
CACHED = ROOT / 'point32_direct_fixed3_v1'
IDS = ('wJPptuhclZmKAEWs', 'wJPrpFsThWAcBQIw', 'wJQTclrlOLEYYfTB')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write(chr(10))


def main():
    destination = FRESH / 'caption_control_v1'
    if destination.exists():
        raise ValueError('New independent output required')
    libraries = {name: sha(CODE / name) for name in
        ('ms_landmarks.py', 'motionscript_v2/landmarks.py', 'motionscript_v2/landmark_events.py', 'motionscript_v2/chunks.py')}
    destination.mkdir()
    cases = []
    for sid in IDS:
        fresh, fm, fh = load_bundle(FRESH / 'raw' / sid)
        cached, cm, ch = load_bundle(CACHED / 'raw' / sid)
        if fm['source'] != 'nlf_fresh_detector_rgb_landmarks' or not fm['no_external_boxes_supplied']:
            raise ValueError('Require genuinely fresh detector32 source')
        for key in ('video', 'model', 'spec'):
            if fm['source_sha256'][key] != cm['source_sha256'][key]:
                raise ValueError('Different source ' + key)
        if not np.array_equal(fresh['names'], cached['names']) or not np.array_equal(fresh['timestamps'], cached['timestamps']):
            raise ValueError('Different same32 query or source PTS')
        if fm['media_bounds'] != cm['media_bounds']:
            raise ValueError('Different actual media windows')
        case_dir = destination / sid
        case_dir.mkdir()
        runs = []
        for source_name, raw, manifest, manifest_hash, path in (
            ('fresh', fresh, fm, fh, FRESH / 'raw' / sid), ('cached', cached, cm, ch, CACHED / 'raw' / sid)):
            source_hash = sha(path / 'raw.npz')
            pixels = raw['points2d_px']
            width, height = manifest['frame_size']
            necessary = raw['valid'] & np.isfinite(pixels).all(-1) & (pixels[:, :, 0] >= 0) & (pixels[:, :, 0] < width) & (pixels[:, :, 1] >= 0) & (pixels[:, :, 1] < height)
            np.savez_compressed(case_dir / (source_name + '_source_masks.npz'), original_numeric_valid=raw['valid'],
                native_points2d_px=pixels, in_frame_necessary_support=necessary,
                timestamps=raw['timestamps'], names=raw['names'], frame_size=np.asarray([width, height]))
            for mode, mask in (('numeric', raw['valid']), ('necessary', necessary)):
                selected = dict(raw)
                selected['valid'] = mask
                arrays, metadata = preprocess_landmarks(selected, fps=20., max_gap_seconds=0., smooth_seconds=.025, max_point_uncertainty=None)
                source_time = raw['timestamps']
                right = np.clip(np.searchsorted(source_time, arrays['time']), 0, len(source_time) - 1)
                exact = np.isclose(source_time[right], arrays['time'], atol=1e-9, rtol=0)
                left = np.where(exact, right, np.maximum(0, right - 1))
                if (arrays['point_valid'] & ~(mask[left] & mask[right])).any():
                    raise ValueError('Rejected raw points were filled across a gap')
                if np.isfinite(arrays['points_m'][~arrays['point_valid']]).any():
                    raise ValueError('Missing coordinates not retained')
                signals, signal_metadata = measure_landmark_signals(arrays)
                arrays.update(signals)
                events = extract_landmark_events(arrays, min_duration_seconds=.15, min_angle_change_degrees=20.)
                chunks = []
                for first, last in ((0., 2.), (2., 5.0625)):
                    sliced, support = slice_landmarks(arrays, first, last)
                    local = extract_landmark_events(sliced, min_duration_seconds=.15, min_angle_change_degrees=20.)
                    chunks.append(dict(window=dict(start=first, end=last), support=support, events=local,
                        evidence_caption=render_landmark_caption(local), caption_is_training_target=False))
                for event in events:
                    evidence = event['evidence']
                    indices = evidence['required_point_indices']
                    start, stop = evidence['sample_start'], evidence['sample_end'] + 1
                    if not arrays['point_valid'][start:stop][:, indices].all():
                        raise ValueError('Event depends on missing point')
                if source_name == 'cached':
                    prior_path = CACHED / (('describe/whole/' + sid + '/report.json') if mode == 'numeric' else ('inframe_control_v1/' + sid + '/report.json'))
                    prior = json.loads(prior_path.read_text())
                    if prior['events'] != events:
                        raise ValueError('Existing cached control no longer matches immutable operators')
                foot_count = sum('foot' in event['body_part'] for event in events)
                if sid == 'wJPrpFsThWAcBQIw' and mode == 'necessary':
                    if foot_count or any('foot' in event['body_part'] for chunk in chunks for event in chunk['events']):
                        raise ValueError('Out-of-frame foot event survived necessary-support filter')
                target = case_dir / (source_name + '_' + mode)
                target.mkdir()
                np.savez_compressed(target / 'coords.npz', **arrays)
                report = dict(schema='nlf-fresh-cached32-support-caption-control-v1', id=sid, source_mode=source_name,
                    support_mode=mode, source_manifest_path=str(path / 'manifest.json'), source_manifest_sha256=manifest_hash,
                    source_raw_path=str(path / 'raw.npz'), source_raw_sha256=source_hash,
                    source_sha256=manifest['source_sha256'], numeric_valid_pointframes=int(raw['valid'].sum()),
                    selected_support_pointframes=int(mask.sum()), excluded_pointframes=int((raw['valid'] & ~mask).sum()),
                    excluded_by_point={str(name): int((raw['valid'][:, j] & ~mask[:, j]).sum()) for j, name in enumerate(raw['names'])},
                    coords_sha256=sha(target / 'coords.npz'), metadata=metadata, continuous_signals=signal_metadata,
                    events=events, whole_event_count=len(events), foot_event_count=foot_count,
                    event_counts_by_body_part=dict(collections.Counter(event['body_part'] for event in events)),
                    evidence_caption=render_landmark_caption(events), chunks=chunks,
                    preprocessing_config=dict(fps=20., max_gap_seconds=0., smooth_seconds=.025, max_point_uncertainty=None),
                    event_config=dict(min_duration_seconds=.15, min_angle_change_degrees=20.),
                    target_support_both_raw_brackets_verified=True, no_missing_point_fill=True,
                    actual_inframe_is_visibility=False, calibrated_geometry_accuracy=False,
                    caption_is_training_target=False, semantic_approval=False, raw_modified=False,
                    implementation_sha256=libraries, helper_sha256=sha(Path(__file__)))
                write_json(target / 'report.json', report)
                write_json(target / 'report.receipt.json', dict(report_sha256=sha(target / 'report.json')))
                runs.append(dict(source=source_name, mode=mode, report_path=str(target / 'report.json'),
                    report_sha256=sha(target / 'report.json'), coords_path=str(target / 'coords.npz'),
                    coords_sha256=sha(target / 'coords.npz'), source_manifest_sha256=manifest_hash,
                    source_raw_sha256=source_hash, numeric_valid_pointframes=report['numeric_valid_pointframes'],
                    selected_support_pointframes=report['selected_support_pointframes'],
                    excluded_pointframes=report['excluded_pointframes'], whole_event_count=len(events),
                    foot_event_count=foot_count, window_event_counts=[len(chunk['events']) for chunk in chunks],
                    event_counts_by_body_part=report['event_counts_by_body_part']))
            if sha(path / 'raw.npz') != source_hash:
                raise ValueError('Raw source changed')
        summary = dict(id=sid, masks=[dict(path=str(case_dir / (name + '_source_masks.npz')),
            sha256=sha(case_dir / (name + '_source_masks.npz'))) for name in ('fresh', 'cached')], runs=runs,
            same_video_model_spec_names_time_verified=True, masks_are_calibrated_visibility=False,
            out_of_frame_foot_events_blocked=True, inframe_occlusion_and_heel_doubts_remain=True,
            caption_accuracy_advantage_established=False)
        cases.append(summary)
        print(json.dumps(dict(id=sid, controls=[dict(source=run['source'], mask=run['mode'],
            events=run['whole_event_count'], feet=run['foot_event_count']) for run in runs])), flush=True)
    for name, digest in libraries.items():
        if sha(CODE / name) != digest:
            raise ValueError('Immutable operator code changed')
    terminal = dict(schema='nlf-fresh-detector32-caption-support-control-terminal-v1', complete=True,
        fixed3_count=3, support_control_runs=12, cases=cases, immutable_operator_sha256=libraries,
        helper_path=str(Path(__file__)), helper_sha256=sha(Path(__file__)),
        same_windows_every_control=[dict(start=0., end=2.), dict(start=2., end=5.0625)],
        necessary_support_not_visibility=True, missing_points_not_interpolated_across_rejection=True,
        old_cached_raw_or_reports_modified=False, fresh_raw_or_producer_modified=False,
        redshirt_outside_foot_events_blocked=True, caption_accuracy_advantage_established=False,
        semantic_approval=False, training_approved=False, gpu_used=False)
    write_json(destination / 'terminal_receipt.json', terminal)


if __name__ == '__main__':
    main()
