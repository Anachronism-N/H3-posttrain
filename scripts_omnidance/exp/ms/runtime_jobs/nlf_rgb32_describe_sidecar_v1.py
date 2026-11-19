"""Run immutable generic describe and bind fresh detector provenance separately."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from ms_landmarks import load_bundle

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
OUT = ROOT / 'diff/logs_omnidance/exp/MS/nlf_rgb32_fresh_detector_fixed3_v1'
IDS = ('wJPptuhclZmKAEWs', 'wJPrpFsThWAcBQIw', 'wJQTclrlOLEYYfTB')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write(chr(10))


def main():
    destination = OUT / 'immutable_describe_v1'
    if destination.exists():
        raise ValueError('New output directory required; preserve previous controls')
    producer = CODE / 'ms_landmarks.py'
    producer_sha = sha(producer)
    if producer_sha != 'f7b3ee3c1c59b9ac19b96f2a73f3bb18a79d4804e715bc1f7334c3e363c467c1':
        raise ValueError('Immutable original describe source changed')
    destination.mkdir()
    cases = []
    for sid in IDS:
        bundle = OUT / 'raw' / sid
        raw, source, source_manifest_sha = load_bundle(bundle)
        if source['source'] != 'nlf_fresh_detector_rgb_landmarks' or not source['no_external_boxes_supplied']:
            raise ValueError('Actual fresh detector source required')
        rows = []
        for mode in ('whole', 'fixed2s'):
            path = destination / sid / mode
            argv = [sys.executable, '-B', str(producer), 'describe', '--bundle', str(bundle), '--out', str(path)]
            if mode == 'fixed2s':
                argv += ['--chunk-seconds', '2', '--min-tail-seconds', '1.5']
            completed = subprocess.run(argv, check=True, capture_output=True, text=True)
            report = json.loads((path / 'report.json').read_text())
            receipt = json.loads((path / 'report.receipt.json').read_text())
            report_sha = sha(path / 'report.json')
            if receipt['report_sha256'] != report_sha or report['coords_sha256'] != sha(path / 'coords.npz'):
                raise ValueError('Describe output receipt mismatch')
            if report['source_manifest_sha256'] != source_manifest_sha or report['raw_sha256'] != source['raw_sha256']:
                raise ValueError('Generic report does not bind actual fresh bundle')
            # Generic source string is the immutable producer's declaration.
            # Actual detector provenance remains exclusively in this sidecar.
            if report['source'] != 'nlf_direct_rgb_landmarks' or report['semantic_approval'] or report['new_information_accuracy_validated']:
                raise ValueError('Unexpected generic report source/approval')
            control_path = OUT / 'caption_control_v1' / sid / 'fresh_numeric/report.json'
            control = json.loads(control_path.read_text())
            if report['events'] != control['events']:
                raise ValueError('CLI candidate events differ from same-source numeric control')
            with np.load(path / 'coords.npz', allow_pickle=False) as coords, np.load(control_path.parent / 'coords.npz', allow_pickle=False) as baseline:
                if not np.array_equal(coords['points_m'], baseline['points_m'], equal_nan=True) or not np.array_equal(coords['point_valid'], baseline['point_valid']):
                    raise ValueError('CLI geometry differs from no-gap control')
            windows = [chunk['window'] for chunk in report.get('chunks', [])]
            if mode == 'fixed2s' and windows != [dict(start=0., end=2.), dict(start=2., end=5.0625)]:
                raise ValueError('Actual fixed2s windows changed')
            sidecar = dict(schema='nlf-fresh32-immutable-generic-describe-source-sidecar-v1', id=sid, mode=mode,
                actual_geometry_source=source['source'], original_report_declared_source=report['source'],
                generic_report_not_relabelled=True, actual_fresh_detector_used=source['fresh_detector_used'],
                no_external_boxes_supplied=source['no_external_boxes_supplied'],
                source_manifest_path=str(bundle / 'manifest.json'), source_manifest_sha256=source_manifest_sha,
                source_raw_path=str(bundle / 'raw.npz'), source_raw_sha256=source['raw_sha256'],
                actual_source_paths=source['source_paths'], actual_source_sha256=source['source_sha256'],
                original_report_path=str(path / 'report.json'), original_report_sha256=report_sha,
                original_coords_path=str(path / 'coords.npz'), original_coords_sha256=report['coords_sha256'],
                argv=argv, exit_code=completed.returncode, producer_path=str(producer), producer_sha256=producer_sha,
                numeric_control_path=str(control_path), numeric_control_sha256=sha(control_path),
                exact_numeric_control_geometry_and_events_equal=True, event_count=len(report['events']),
                foot_event_count=sum('foot' in event['body_part'] for event in report['events']),
                annotation_windows=windows, window_event_counts=[len(chunk['events']) for chunk in report.get('chunks', [])],
                necessary_support_control_path=str(OUT / 'caption_control_v1' / sid / 'fresh_necessary/report.json'),
                necessary_support_control_sha256=sha(OUT / 'caption_control_v1' / sid / 'fresh_necessary/report.json'),
                numeric_candidates_are_not_visible_facts=True, visibility_verified=False,
                accuracy_approved=False, training_approved=False, old_report_or_source_rewritten=False)
            save(path / 'actual_source_sidecar.json', sidecar)
            rows.append(dict(mode=mode, sidecar_path=str(path / 'actual_source_sidecar.json'), sidecar_sha256=sha(path / 'actual_source_sidecar.json'),
                report_path=str(path / 'report.json'), report_sha256=report_sha,
                coords_path=str(path / 'coords.npz'), coords_sha256=report['coords_sha256'],
                event_count=sidecar['event_count'], foot_event_count=sidecar['foot_event_count'],
                windows=windows, window_event_counts=sidecar['window_event_counts']))
            print(json.dumps(dict(id=sid, mode=mode, exit_code=0, events=len(report['events']), actual_source=source['source'])), flush=True)
        cases.append(dict(id=sid, source_manifest_sha256=source_manifest_sha, descriptions=rows))
    if sha(producer) != producer_sha:
        raise ValueError('Live describe producer changed during processing')
    terminal = dict(schema='nlf-fresh32-immutable-describe-sidecar-terminal-v1', complete=True, fixed3_count=3,
        actual_cli_describe_runs=6, cases=cases, producer_sha256=producer_sha, helper_path=str(Path(__file__)), helper_sha256=sha(Path(__file__)),
        actual_geometry_source='nlf_fresh_detector_rgb_landmarks', original_generic_report_source_preserved=True,
        necessary_control_terminal_path=str(OUT / 'caption_control_v1/terminal_receipt.json'),
        necessary_control_terminal_sha256=sha(OUT / 'caption_control_v1/terminal_receipt.json'),
        full_source_or_raw_modified=False, GPU_used=False, accuracy_advantage_established=False, training_approved=False)
    save(destination / 'terminal_receipt.json', terminal)


if __name__ == '__main__':
    main()
