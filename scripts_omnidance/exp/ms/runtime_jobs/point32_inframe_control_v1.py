import collections
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
CODE = ROOT / 'H3-posttrain/scripts_omnidance/exp/ms'
sys.path.insert(0, str(CODE))
from ms_landmarks import load_bundle
from motionscript_v2.landmarks import preprocess_landmarks
from motionscript_v2.landmark_events import extract_landmark_events, measure_landmark_signals, render_landmark_caption
from motionscript_v2.chunks import slice_landmarks

OUT = ROOT / 'diff/logs_omnidance/exp/MS/point32_direct_fixed3_v1'
IDS = ('wJPptuhclZmKAEWs', 'wJPrpFsThWAcBQIw', 'wJQTclrlOLEYYfTB')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write(chr(10))


def main():
    destination = OUT / 'inframe_control_v1'
    if destination.exists():
        raise ValueError('Fresh independent derived-analysis output required')
    libraries = {str(CODE / name): sha(CODE / name) for name in
        ('ms_landmarks.py', 'motionscript_v2/landmarks.py', 'motionscript_v2/landmark_events.py', 'motionscript_v2/chunks.py')}
    destination.mkdir()
    cases = []
    for sid in IDS:
        bundle = OUT / 'raw' / sid
        raw, source, manifest_sha = load_bundle(bundle)
        original_raw_sha = sha(bundle / 'raw.npz')
        numeric_valid = raw['valid'].copy()
        pixels = raw['points2d_px']
        width, height = source['frame_size']
        finite2d = np.isfinite(pixels).all(axis=2)
        inbounds = ((pixels[..., 0] >= 0) & (pixels[..., 0] < width)
                    & (pixels[..., 1] >= 0) & (pixels[..., 1] < height))
        necessary = numeric_valid & finite2d & inbounds
        original_report_path = OUT / 'describe/whole' / sid / 'report.json'
        original_report = json.loads(original_report_path.read_text())
        original_coords_path = OUT / 'describe/whole' / sid / 'coords.npz'
        assert sha(original_coords_path) == original_report['coords_sha256']
        assert original_report['source_manifest_sha256'] == manifest_sha
        config = dict(fps=20., max_gap_seconds=0., smooth_seconds=.025, max_point_uncertainty=None)
        # No missing source point is bridged. 16->20fps resampling within
        # consecutive supported raw frames remains the immutable operator.
        baseline, _ = preprocess_landmarks(raw, **config)
        baseline_events = extract_landmark_events(baseline, min_duration_seconds=.15, min_angle_change_degrees=20.)
        assert baseline_events == original_report['events'], 'No-gap baseline changed original event control'
        with np.load(original_coords_path, allow_pickle=False) as old:
            assert np.array_equal(old['points_m'], baseline['points_m'], equal_nan=True)
            assert np.array_equal(old['point_valid'], baseline['point_valid'])
        selected = dict(raw)
        selected['valid'] = necessary
        derived, metadata = preprocess_landmarks(selected, **config)
        source_time = raw['timestamps']
        # Independently verify no target sample got support across any rejected
        # raw point: both bracketing raw frames must satisfy the necessary mask.
        right = np.searchsorted(source_time, derived['time'], side='left')
        right = np.clip(right, 0, len(source_time) - 1)
        exact = np.isclose(derived['time'], source_time[right], atol=1e-9, rtol=0)
        left = np.where(exact, right, np.maximum(right - 1, 0))
        bracket_support = necessary[left] & necessary[right]
        assert not (derived['point_valid'] & ~bracket_support).any()
        assert np.isnan(derived['points_m'][~derived['point_valid']]).all()
        signals, signal_metadata = measure_landmark_signals(derived)
        derived.update(signals)
        events = extract_landmark_events(derived, min_duration_seconds=.15, min_angle_change_degrees=20.)
        chunks = []
        for start, end in ((0., 2.), (2., 5.0625)):
            sliced, support = slice_landmarks(derived, start, end)
            local = extract_landmark_events(sliced, min_duration_seconds=.15, min_angle_change_degrees=20.)
            chunks.append(dict(window=dict(start=start, end=end), support=support, events=local,
                evidence_caption=render_landmark_caption(local), caption_is_training_target=False))
        name_counts = {str(name): int((numeric_valid[:, j] & ~necessary[:, j]).sum()) for j, name in enumerate(raw['names'])}
        foot_event_count = sum('foot' in event['body_part'] for event in events)
        if sid == 'wJPrpFsThWAcBQIw':
            foot_ids = [j for j, name in enumerate(raw['names']) if any(suffix in str(name) for suffix in ('heel', 'big_toe', 'little_toe'))]
            assert len(foot_ids) == 6 and not necessary[:, foot_ids].any()
            assert foot_event_count == 0 and all(not any('foot' in event['body_part'] for event in chunk['events']) for chunk in chunks)
            assert not derived['point_valid'][:, foot_ids].any()
        directory = destination / sid
        directory.mkdir()
        np.savez_compressed(directory / 'masks.npz', raw_numeric_valid=numeric_valid,
            finite_native_points2d=finite2d, native_points2d_px=pixels,
            in_frame_bounds=inbounds, in_frame_necessary_support=necessary,
            timestamps=raw['timestamps'], names=raw['names'], frame_size=np.asarray([width, height]))
        np.savez_compressed(directory / 'derived_coords.npz', **derived)
        report = dict(schema='nlf-direct32-inframe-necessary-support-candidates-v1', id=sid,
            mask_definition='raw_numeric_valid AND finite(native_points2d_px) AND 0<=x<width AND 0<=y<height',
            support_semantics='in_frame_necessary_support; necessary only, NOT visibility or calibrated confidence',
            source_manifest_path=str(bundle / 'manifest.json'), source_manifest_sha256=manifest_sha,
            source_raw_path=str(bundle / 'raw.npz'), source_raw_sha256=original_raw_sha,
            original_report_path=str(original_report_path), original_report_sha256=sha(original_report_path),
            masks_path=str(directory / 'masks.npz'), masks_sha256=sha(directory / 'masks.npz'),
            derived_coords_path=str(directory / 'derived_coords.npz'), derived_coords_sha256=sha(directory / 'derived_coords.npz'),
            numeric_valid_point_frames=int(numeric_valid.sum()), necessary_support_point_frames=int(necessary.sum()),
            excluded_point_frames=int((numeric_valid & ~necessary).sum()), excluded_by_point=name_counts,
            original_events=original_report['events'], original_event_count=len(original_report['events']),
            original_no_gap_baseline_exact_equal=True, original_no_gap_coords_exact_equal=True,
            no_rejected_point_gap_fill=True, target_valid_requires_both_raw_brackets=True,
            preprocess_config=config, original_preprocess_max_gap_seconds=.2,
            event_config=dict(min_duration_seconds=.15, min_angle_change_degrees=20.),
            metadata=metadata, continuous_signals=signal_metadata, events=events,
            derived_event_count=len(events), derived_foot_event_count=foot_event_count,
            derived_events_by_body_part=dict(collections.Counter(event['body_part'] for event in events)),
            evidence_caption=render_landmark_caption(events), chunks=chunks,
            implementation_sha256=libraries, helper_sha256=sha(Path(__file__)),
            inframe_occluded_support_remains_uncertain=True, precise_geometry_approved=False,
            caption_is_training_target=False, semantic_approval=False, raw_modified=False)
        write_json(directory / 'report.json', report)
        write_json(directory / 'report.receipt.json', dict(report_sha256=sha(directory / 'report.json')))
        assert sha(bundle / 'raw.npz') == original_raw_sha
        cases.append({key: report[key] for key in ('id', 'source_manifest_path', 'source_manifest_sha256', 'source_raw_path', 'source_raw_sha256',
            'masks_path', 'masks_sha256', 'derived_coords_path', 'derived_coords_sha256', 'numeric_valid_point_frames',
            'necessary_support_point_frames', 'excluded_point_frames', 'excluded_by_point', 'original_event_count',
            'derived_event_count', 'derived_foot_event_count', 'derived_events_by_body_part', 'original_no_gap_baseline_exact_equal',
            'no_rejected_point_gap_fill')})
        cases[-1].update(report_path=str(directory / 'report.json'), report_sha256=sha(directory / 'report.json'),
            window_event_counts=[len(chunk['events']) for chunk in chunks])
        print(json.dumps(dict(id=sid, excluded_point_frames=report['excluded_point_frames'], original_events=report['original_event_count'],
            derived_events=report['derived_event_count'], derived_foot_events=foot_event_count)), flush=True)
    for path, digest in libraries.items():
        assert sha(path) == digest, 'Immutable operator code changed'
    terminal = dict(schema='nlf-direct32-inframe-control-terminal-v1', complete=True, fixed3_count=3,
        cases=cases, mask_is_visibility_estimate=False, raw_geometry_reextracted=False,
        original_events_preserved=True, immutable_operators_used=True, rejected_points_not_gap_filled=True,
        redshirt_outside_foot_events_blocked=True, inframe_occlusion_and_heel_location_not_resolved=True,
        geometry_or_caption_accuracy_advantage_established=False, calibrated_accuracy_claim=False,
        training_approved=False, source_modified=False, gpu_used=False, implementation_sha256=libraries,
        helper_path=str(Path(__file__)), helper_sha256=sha(Path(__file__)))
    write_json(destination / 'terminal_receipt.json', terminal)


if __name__ == '__main__':
    main()
