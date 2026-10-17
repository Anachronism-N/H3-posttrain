"""Describe fixed-case motion after text removal; no dance, beat or caption accuracy score."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from music_counterfactual_eval import (GROUPS, cache_path, cases, load_verified,
                                      motion_series, source_fingerprint)
from music_generation_validate import atomic_write_new, sha256

EMPTY_NATIVE = 'empty_native_h3_v1'
PAIRS = {
    EMPTY_NATIVE: 'native_cleanpad',
    'empty_musicfm_scale_d6000_l0_s1875_v4': 'musicfm_scale_d6000_l0_s1875_v4',
    'empty_musicfm_scale_d6000_l1_s1875_v4': 'musicfm_scale_d6000_l1_s1875_v4',
}
TAGS = [*PAIRS, *[t for t in PAIRS.values() if t != 'native_cleanpad']]


def motion_statistics(data):
    result = {}
    h = float(data['h'])
    window = (np.arange(len(data['xy'])) / float(data['fps']) >= 1) & (
        np.arange(len(data['xy'])) / float(data['fps']) < 5)
    for group in ('body', 'arms', 'legs'):
        _, speed, _ = motion_series(data, *GROUPS[group])
        valid = np.isfinite(speed)
        adequate = float(valid.mean()) >= .7
        start, end = GROUPS[group]
        points = data['xy'][window, start:end]
        observed = data['valid'][window, start:end] & np.isfinite(points).all(-1)
        spans = []
        for joint in range(end - start):
            mask = observed[:, joint]
            if mask.mean() >= .7:
                low, high = np.quantile(points[mask, joint], [.05, .95], axis=0)
                spans.append(float(np.linalg.norm(high - low) / h))
        result[group] = dict(usable_speed_fraction=float(valid.mean()),
                             mean_speed_heights_per_second=float(speed[valid].mean()) if adequate else None,
                             median_speed_heights_per_second=float(np.median(speed[valid])) if adequate else None,
                             mean_joint_p05_p95_span_heights=float(np.mean(spans)) if spans else None,
                             span_usable_joints=len(spans))
    return result


def pose_distance(first, second):
    if float(first['fps']) != float(second['fps']) or len(first['xy']) != len(second['xy']):
        raise ValueError('Pose distance requires matching generated native frame grids')
    selected = (np.arange(len(first['xy'])) / float(first['fps']) >= 1) & (
        np.arange(len(first['xy'])) / float(first['fps']) < 5)
    a = first['xy'][selected, :18] / float(first['h'])
    b = second['xy'][selected, :18] / float(second['h'])
    va, vb = first['valid'][selected, :18], second['valid'][selected, :18]
    pelvis_valid = va[:, [8, 11]].all(1) & vb[:, [8, 11]].all(1)
    a = a - a[:, [8, 11]].mean(1)[:, None]
    b = b - b[:, [8, 11]].mean(1)[:, None]
    mask = va & vb & pelvis_valid[:, None] & np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
    adequate = float(mask.mean()) >= .7
    distances = np.linalg.norm(a - b, axis=-1)
    return dict(valid_pair_fraction=float(mask.mean()),
                pelvis_centered_pose_rms_heights=float(np.sqrt(np.mean(distances[mask] ** 2))) if adequate else None,
                scope='Same-time pose difference, not correct music response, beat synchrony or dance quality.')


def captions(exp_root, selected):
    path = exp_root / 'E10/val47_motion_captions.csv'
    with path.open(newline='') as stream:
        descriptions = {r['sample_id']: r['motion_caption'] for r in csv.DictReader(stream)}
    genre_path = exp_root / 'MS/vlm/eval3_32b.jsonl'
    genres = {r['sid']: r['label'] for r in
              (json.loads(line) for line in genre_path.read_text().splitlines() if line.strip())}
    result = {}
    for entry in selected:
        sid = Path(entry['correct_audio']).stem
        if sid in result:
            continue
        ms_path = exp_root / 'MS/ms_orig_eval3' / (sid + '.json')
        ms = json.loads(ms_path.read_text()) if ms_path.is_file() else None
        result[sid] = dict(audio_path=entry['correct_audio'],
                           video_source_caption=descriptions.get(sid),
                           video_source_caption_kind='E10 video-observed VLM motion caption, NOT original MotionScript',
                           original_motionscript=' '.join(x for x in ms['ms'] if x) if ms else None,
                           original_motionscript_path=str(ms_path) if ms else None,
                           original_motionscript_sha256=sha256(ms_path) if ms else None,
                           video_observed_genre=genres.get(sid, {}).get('genre'),
                           genre_kind='Video-observed VLM label, not music genre or human ground truth',
                           missing_original_motionscript=ms is None)
    return dict(audio_sources=result, vlm_motion_csv_sha256=sha256(path),
                genre_file_sha256=sha256(genre_path),
                scope='Captions describe a dance recorded with that music, not its unique required choreography. Holdout audio IDs were not used for CLIP training.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp_root', type=Path, required=True)
    parser.add_argument('--pose_root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.out.parent.is_dir() or args.out.exists():
        parser.error('Require existing output parent and a new report path')
    selected = cases(args.exp_root, tags=TAGS)
    data_by_key, rows = {}, []
    for entry in selected:
        fingerprint = source_fingerprint(entry)
        path = cache_path(args.pose_root, entry)
        data, _ = load_verified(path, entry, fingerprint)
        data_by_key[entry['tag'], entry['condition'], entry['sample_id']] = data
        rows.append(dict(**entry, source_sha256=fingerprint['sha256'],
                         pose_cache_sha256=sha256(path),
                         tracked_fraction=float(data['tracked'].mean()),
                         motion=motion_statistics(data)))
    index = {(r['tag'], r['condition'], r['sample_id']): r for r in rows}
    text_pairs, swap_pairs = [], []
    ids = list(dict.fromkeys(r['sample_id'] for r in rows))
    for empty, generic in PAIRS.items():
        for condition in ('joint', 'aswap'):
            for sid in ids:
                e = index[empty, condition, sid]
                g = index[generic, condition, sid]
                ratios = {}
                for group in ('body', 'arms', 'legs'):
                    a = e['motion'][group]['mean_speed_heights_per_second']
                    b = g['motion'][group]['mean_speed_heights_per_second']
                    ratios[group] = a / b if a is not None and b is not None and b > 1e-12 else None
                text_pairs.append(dict(empty_tag=empty, generic_tag=generic,
                                       condition=condition, sample_id=sid,
                                       empty_over_generic_speed=ratios,
                                       **pose_distance(data_by_key[empty, condition, sid],
                                                       data_by_key[generic, condition, sid])))
        for sid in ids:
            swap_pairs.append(dict(tag=empty, sample_id=sid,
                                   **pose_distance(data_by_key[empty, 'joint', sid],
                                                   data_by_key[empty, 'aswap', sid])))
            swap_pairs.append(dict(tag=generic, sample_id=sid,
                                   **pose_distance(data_by_key[generic, 'joint', sid],
                                                   data_by_key[generic, 'aswap', sid])))
    summaries = {}
    for tag in (*TAGS, 'native_cleanpad', 'GT'):
        subset = [r for r in rows if r['tag'] == tag]
        summaries[tag] = {'videos': len(subset), 'groups': {}}
        for group in ('body', 'arms', 'legs'):
            values = [r['motion'][group]['mean_speed_heights_per_second'] for r in subset]
            values = [v for v in values if v is not None]
            summaries[tag]['groups'][group] = dict(mean_speed=float(np.mean(values)) if values else None,
                                                   valid_videos=len(values))
    result = dict(complete=True, validated_videos=len(rows), generated_videos=36, GT_controls=3,
                  protocol=dict(window_seconds=[1., 5.],
                                speed_unit='video-frame-height per second',
                                valid_threshold=.3, minimum_speed_coverage=.7,
                                no_fill_or_cross_missing_derivatives=True,
                                span='Per-joint 5th-to-95th percentile XY span, divided by image height'),
                  cases=rows, text_removal_pairs=text_pairs, music_swap_pairs=swap_pairs,
                  summaries=summaries, caption_sources=captions(args.exp_root, selected),
                  scope='Motion amount and same-time pose difference only. Not a caption, dance or music accuracy score; no significance or IID assumption for three fixed cases.')
    atomic_write_new(args.out, result)
    assert json.loads(args.out.read_text()) == result
    print(json.dumps(dict(complete=True, validated_videos=len(rows), summary=summaries,
                          text_removal_pairs=text_pairs, music_swap_pairs=swap_pairs), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
