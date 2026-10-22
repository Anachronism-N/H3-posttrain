"""Cross-check the completed fixed 18 direct-native student videos; not a quality score."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from music_clip_bridge import load_audio
from music_generation_validate import atomic_write_new
from music_text_condition import file_sha256, tensor_sha256


def audit(queue):
    queue = Path(queue).resolve(strict=True)
    plan = json.loads((queue / 'plan.json').read_text())
    finished = json.loads((queue / 'finished.json').read_text())
    if (not finished.get('complete') or finished.get('expected') != 18
            or len(finished.get('audited_configurations', [])) != 6):
        raise ValueError('Require all six completed three-case media audits')
    expected = {(model, condition) for model in (0, 1)
                for condition in ('joint', 'aswap', 'zeroM')}
    configurations = finished['audited_configurations']
    if {(s['model'], s['condition']) for s in configurations} != expected:
        raise ValueError('Missing, extra or duplicate student configurations')
    rows, reports, student_waves, native_waves = {}, [], {}, {}
    for spec in configurations:
        selected = next(s for s in plan['specifications']
                        if (s['model'], s['condition']) == (spec['model'], spec['condition']))
        if any(spec[key] != selected[key] for key in ('out', 'audit', 'checkpoint', 'manifest', 'host')):
            raise ValueError('Completed configuration differs from the selected plan')
        report_path = Path(spec['audit'])
        report = json.loads(report_path.read_text())
        if (spec.get('exit_code') != 0 or spec.get('complete') is not True
                or report.get('complete') is not True or report.get('expected') != 3
                or report.get('passed') != 3
                or report.get('complete_selected_audio_video_decode') is not True
                or report.get('musical_quality_accepted') is not False
                or Path(report['directory']).resolve() != Path(spec['out']).resolve()
                or len(report['rows']) != 3):
            raise ValueError('Incomplete independent three-case media report')
        reports.append(dict(model=spec['model'], condition=spec['condition'],
                            path=str(report_path), sha256=file_sha256(report_path)))
        generation_plan = report['plan']
        if (generation_plan['teacher_identity'] or not generation_plan['no_text_encoder']
                or generation_plan['student_zero_input'] != (spec['condition'] == 'zeroM')
                or generation_plan['checkpoint'] != spec['checkpoint']
                or generation_plan['music_mode'] != 'ref'
                or generation_plan['output_audio'] != 'generated'):
            raise ValueError('Student checkpoint/native-audio/zero-input plan mismatch')
        for row in report['rows']:
            case, inputs, native = row['case'], row['input'], row['audit']
            key = (spec['model'], spec['condition'], case['sample_id'])
            if key in rows or row['integrity'].get('full_video_audio_decode') is not True:
                raise ValueError('Duplicate or undecoded student video')
            video, log = Path(row['video_path']), Path(row['log_path'])
            if file_sha256(video) != row['video_sha256'] or file_sha256(log) != row['log_sha256']:
                raise ValueError('Video or sampler log changed since independent audit')
            saved = json.loads(video.with_suffix('.audit.json').read_text())
            if saved['input'] != inputs or saved['audit'] != native or saved['video_sha256'] != row['video_sha256']:
                raise ValueError('Saved input record differs from independently audited media')
            audio_path = Path(case['audio_path'])
            if file_sha256(audio_path) != case['audio_sha256']:
                raise ValueError('Input music source changed')
            if audio_path not in student_waves:
                student_waves[audio_path], _ = load_audio(audio_path.stem, audio_root=audio_path.parent)
            wave = student_waves[audio_path]
            expected_wave = torch.zeros_like(wave) if spec['condition'] == 'zeroM' else wave
            if tensor_sha256(expected_wave) != inputs['student_waveform_sha256']:
                raise ValueError('Student waveform is not the exact selected music/zero input')
            if audio_path not in native_waves:
                import librosa
                values, rate = librosa.load(str(audio_path), sr=inputs['native_sample_rate'],
                                           mono=False, duration=124 / 24)
                native_wave = torch.from_numpy(values).float()
                if native_wave.ndim == 1:
                    native_wave = native_wave.unsqueeze(0).repeat(2, 1)
                native_waves[audio_path] = (native_wave, rate)
            native_wave, rate = native_waves[audio_path]
            if (int(rate) != inputs['native_sample_rate']
                    or tensor_sha256(native_wave) != inputs['native_waveform_sha256']
                    or native['waveform_sha256'] != inputs['native_waveform_sha256']
                    or saved['mux_audio_sha256'] != native['generated_audio_sha256']):
                raise ValueError('Native music or preserved generated audio source differs')
            rows[key] = row
    ids = {key[2] for key in rows}
    if len(rows) != 18 or len(ids) != 3:
        raise ValueError('Require exactly the original three people in all six configurations')
    checks = []
    for sid in sorted(ids):
        reference = rows[(0, 'joint', sid)]
        for model in (0, 1):
            joint, zero, swap = (rows[(model, condition, sid)]
                                 for condition in ('joint', 'zeroM', 'aswap'))
            for row in (joint, zero, swap):
                if (any(row['case'][key] != reference['case'][key] for key in
                        ('image_path', 'image_sha256', 'prefix_path', 'prefix_file_sha256',
                         'prefix_sha256', 'prefix_tags_sha256', 'prepared_image_sha256'))
                        or row['audit']['sampling'] != reference['audit']['sampling']
                        or row['audit']['visual_reference'] != reference['audit']['visual_reference']
                        or row['audit']['scheduler_updates'] != dict(video=50, audio=50)):
                    raise ValueError('Image, P/tags, visual VAE condition or sampling changed')
            for row in (joint, zero):
                if (row['input']['native_waveform_sha256'] != reference['input']['native_waveform_sha256']
                        or row['audit']['reference'] != reference['audit']['reference']):
                    raise ValueError('Joint/zeroM did not preserve the same real native music')
            if (swap['case']['original_audio_sha256'] != joint['case']['audio_sha256']
                    or swap['case']['audio_sha256'] == joint['case']['audio_sha256']
                    or swap['input']['native_waveform_sha256'] == joint['input']['native_waveform_sha256']
                    or swap['input']['student_waveform_sha256'] == joint['input']['student_waveform_sha256']
                    or zero['input']['student_waveform_sha256'] == joint['input']['student_waveform_sha256']):
                raise ValueError('Swapped/zero music counterfactual did not actually change input')
            checks.append(dict(model=model, sample_id=sid, same_image_prefix_tags_visual_sampling=True,
                               zeroM_exact_zero_and_native_real_music_preserved=True,
                               aswap_changes_both_student_and_native_music=True))
    return dict(complete=True, expected=18, fixed_people=sorted(ids), reports=reports, checks=checks,
                musical_quality_accepted=False, human_review_required=True,
                scope='Cross-configuration input/source/media integrity, not M-only swap causality or dance quality')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not args.out.parent.is_dir():
        parser.error('Require a new report path and an existing parent')
    result = audit(args.queue)
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    atomic_write_new(args.out, result)
    print(json.dumps(dict(complete=True, expected=18, out=str(args.out), human_review_required=True)), flush=True)


if __name__ == '__main__':
    main()
