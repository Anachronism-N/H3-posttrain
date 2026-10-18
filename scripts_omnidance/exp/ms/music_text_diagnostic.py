"""Audit direct inference of an unchanged native-H candidate; quality gates stay recorded."""
import argparse
import json
from pathlib import Path
import re

from music_generation_validate import (
    atomic_write_new, fixed_cases, probe_video, sha256, signature, successful_log,
)


DEFAULT_TAG = 'text_Hclip_realneg_diag_v1'
ZERO_TAG = 'text_Hclip_realneg_diag_zero_v1'
EXPECTED_SHA = '895868746f3fdec2b58acdae7fe1345d9c5670195e1850aef714a6930c301f45'


def one_record(text, prefix):
    records = re.findall(r'^' + re.escape(prefix) + r'(.+)$', text, flags=re.MULTILINE)
    if len(records) != 1:
        raise ValueError('Require exactly one audit record: ' + prefix)
    return json.loads(records[0])


def diagnostic_plan(exp, out, jobs):
    checkpoint = exp / 'MS/music_text_Hclip_realneg_d1800_s1875_v2/last.pt'
    summary = json.loads((checkpoint.parent / 'finished.json').read_text())
    if (not summary['complete'] or summary['checkpoint_sha256'] != EXPECTED_SHA
            or summary['musical_dependency_gate'] is not False):
        raise ValueError('Candidate does not match the previously audited diagnostic selection')
    cases = fixed_cases(exp)
    result = dict(
        experiment='Unchanged previous native-H sequence CLIP model, diagnostic inference only',
        diagnostic_only=True, quality_gate_passed=False, previous_quality_gate_preserved=True,
        authorization='User explicitly requested post-training video inference without prior length calibration',
        calibration_or_new_training=False, checkpoint=str(checkpoint), checkpoint_sha256=EXPECTED_SHA,
        checkpoint_summary_sha256=sha256(checkpoint.parent / 'finished.json'),
        checkpoint_sha256_source='previous_independent_checkpoint_audit_and_finished_report; full_content_rechecked_in_final_video_audit',
        original_representation=dict(correct_beats_wrong_fraction=summary['final']['correct_beats_wrong_fraction'],
            correct_minus_wrong_score=summary['final']['correct_minus_wrong_score'],
            length_mae=summary['final_regression']['length_mae']),
        source_people=[row['sample_id'] for row in cases[:3]],
        groups=[dict(tag=DEFAULT_TAG, conditions=['joint', 'aswap'], expected=6, job_handle=jobs[0]),
                dict(tag=ZERO_TAG, conditions=['joint'], expected=3, job_handle=jobs[1])],
        native_reference_and_retake_audio='same_fixed_silence', semantic_prompt=' ',
        music_route='Full trained adapter raw H tail -> original DiT text condition_proj and token_refiner',
        music_tokens_reencoded_by_native_text_encoder=False, teacher_caption_at_inference=False,
        predicted_length_not_ground_truth=True, input_backbone='public pretrained MusicFM, frozen during adapter training',
        settings=dict(height=1280, width=704, fps=24, frames=124, steps=50, shift=12,
                      audio_shift=3, seed=42, mode='bi', cfg=1, audio_tail_mode='repeat_clean'),
        resource_use='Six nodes, one admitted H3 cold loader per node; three normal and three zero queues',
        previous_MSE_baseline='text_trained1800_v2 / text_trained1800_zero_v2, existing videos reused',
        scope='No claim of completed musical quality or of passing the original length gate')
    atomic_write_new(out, result)
    print(json.dumps(dict(event='diagnostic_plan_saved', diagnostic_only=True, videos=9, out=str(out))), flush=True)


def audit(exp, out):
    import decord
    import numpy as np
    from PIL import Image, ImageDraw

    checkpoint = exp / 'MS/music_text_Hclip_realneg_d1800_s1875_v2/last.pt'
    summary_path = checkpoint.parent / 'finished.json'
    summary = json.loads(summary_path.read_text())
    if (not summary['complete'] or summary['checkpoint_sha256'] != EXPECTED_SHA
            or summary['musical_dependency_gate'] is not False or sha256(checkpoint) != EXPECTED_SHA):
        raise ValueError('Original diagnostic checkpoint content or failed quality-gate record changed')
    cases = fixed_cases(exp)
    rows, mosaics, native_hashes, pending_groups, pending_mosaics = [], [], [], [], []
    specifications = [(DEFAULT_TAG, 'joint'), (DEFAULT_TAG, 'aswap'), (ZERO_TAG, 'joint')]
    for tag, condition in specifications:
        directory = exp / 'H3AR' / ('val_musicreplace_' + tag)
        jobs = json.loads((directory / 'jobs.json').read_text())
        if not jobs['all_sampler_jobs_exit_zero']:
            raise ValueError('Incomplete diagnostic sampler jobs')
        grid = Image.new('RGB', (176 * 7, 348 * 3), (245, 245, 245))
        draw = ImageDraw.Draw(grid)
        for index, case in enumerate(row for row in cases if row['condition'] == condition):
            evidence = successful_log(directory / 'logs', case, require_replacement=False, require_empty_prompt=True)
            if evidence is None:
                raise ValueError('Missing exact successful log for ' + case['filename'])
            log_path, log_sha, log_signature, _ = evidence
            text = log_path.read_text()
            inputs = one_record(text, '[h3ar] input_conditions ')
            deployment = one_record(text, '[music text deployment] ')
            native = one_record(text, '[music text native isolation] ')
            zero = tag == ZERO_TAG
            if (inputs.get('music_text_ckpt') != str(checkpoint) or inputs.get('music_text_zero_input') != zero
                    or inputs.get('text_hidden_oracle') is not False or inputs.get('legacy_clip_complete_chain') is not False
                    or inputs.get('native_audio_condition') != 'fixed_silence'
                    or inputs.get('waveform_loader') != 'music_clip_bridge.load_audio'):
                raise ValueError('Diagnostic input route mismatch')
            shape = deployment.get('tail_shape', [])
            if (deployment.get('checkpoint_sha256') != EXPECTED_SHA or len(shape) != 2 or shape[1] != 5120
                    or not 1 <= shape[0] <= 1423 or shape[0] != deployment.get('predicted_length')
                    or deployment.get('final_adapter_projection_executed') is not True
                    or deployment.get('uses_ground_truth_caption') is not False
                    or deployment.get('legacy_clip_complete_chain') is not False):
                raise ValueError('Diagnostic deployed raw-H mapping mismatch')
            if native.get('native_music_bypass') is not False or native.get('all_target_audio_clean') is not True:
                raise ValueError('Real music entered a native audio condition')
            video = directory / condition / case['filename']
            before = signature(video)
            integrity = probe_video(video)
            reader = decord.VideoReader(str(video), num_threads=1)
            with Image.open(inputs['image_path']) as image:
                reference = image.convert('RGB')
            samples = [reference] + [Image.fromarray(reader[frame].asnumpy()) for frame in (0, 24, 48, 72, 96, 123)]
            for column, image in enumerate(samples):
                grid.paste(image.resize((176, 320)), (column * 176, index * 348 + 28))
                label = case['sample_id'] if column == 0 else str((0, 24, 48, 72, 96, 123)[column - 1] / 24) + 's'
                draw.text((column * 176 + 4, index * 348 + 8), label, fill='black')
            row = dict(tag=tag, condition=condition, sample_id=case['sample_id'], audio_path=case['audio_path'],
                       video_path=str(video), video_sha256=sha256(video), log_path=str(log_path), log_sha256=log_sha,
                       inputs=inputs, deployment=deployment, native_audio=native, integrity=integrity,
                       diagnostic_only=True, quality_gate_passed=False,
                       audio_mux_match_checked=False,
                       audio_mux_scope='Selected audio stream fully decoded; input music and mux source traced in sampler code, not an independent source-audio equality test')
            if signature(video) != before or signature(log_path) != log_signature:
                raise ValueError('Completed diagnostic artifact changed during audit')
            native_hashes.append(native)
            rows.append(row)
        mosaic = directory / (condition + '_mosaic.png')
        if mosaic.exists():
            raise FileExistsError('Refuse existing diagnostic mosaic: ' + str(mosaic))
        pending_mosaics.append((mosaic, grid))
        mosaics.append(str(mosaic))
        group_rows = [row for row in rows if row['tag'] == tag]
        marker = directory / ('all6_integrity.json' if condition == 'aswap' else 'joint_integrity.json')
        pending_groups.append((marker, dict(complete=True, expected=len(group_rows), passed=len(group_rows),
                                            rows=group_rows, mosaic=str(mosaic), diagnostic_only=True,
                                            quality_gate_passed=False, final_diagnostic_report=str(out))))
        print(json.dumps(dict(event='diagnostic_group_decoded', tag=tag, condition=condition,
                              passed=3, mosaic=str(mosaic))), flush=True)
    if len(rows) != 9 or any(len({row.get(key) for row in native_hashes}) != 1 or
                            any(not row.get(key) for row in native_hashes)
                            for key in ('waveform_sha256', 'reference_sha256', 'retake_sha256', 'mask_sha256')):
        raise ValueError('All nine native audio conditions must be identical')
    by_key = {(row['condition'], row['sample_id'], row['tag']): row for row in rows}
    pixel_pairs = []
    for case in cases[:3]:
        sid = case['sample_id']
        selected = [by_key['joint', sid, DEFAULT_TAG], by_key['aswap', sid, DEFAULT_TAG], by_key['joint', sid, ZERO_TAG]]
        readers = [decord.VideoReader(row['video_path'], num_threads=1) for row in selected]
        differences = [[], []]
        motion, previous = [[], [], []], [None, None, None]
        for frame in range(124):
            values = [reader[frame].asnumpy().astype(np.float32) / 255 for reader in readers]
            for j in (0, 1):
                differences[j].append(float(np.abs(values[0] - values[j + 1]).mean()))
            for j, value in enumerate(values):
                if previous[j] is not None:
                    motion[j].append(float(np.abs(value - previous[j]).mean()))
                previous[j] = value
        pixel_pairs.append(dict(sample_id=sid, original_swapped_rgb_mae=float(np.mean(differences[0])),
                               original_zero_rgb_mae=float(np.mean(differences[1])),
                               adjacent_frame_rgb_mae=dict(original=float(np.mean(motion[0])),
                                   swapped=float(np.mean(motion[1])), zero=float(np.mean(motion[2]))),
                               predicted_lengths=[row['deployment']['predicted_length'] for row in selected]))
    result = dict(complete=True, expected=9, passed=9, rows=rows, mosaics=mosaics,
                  pixel_pairs=pixel_pairs, all_native_audio_hashes_identical=True,
                  checkpoint_sha256=EXPECTED_SHA, diagnostic_only=True, quality_gate_passed=False,
                  original_checkpoint_summary_sha256=sha256(summary_path), previous_quality_gate_independently_rechecked=True,
                  training_or_length_calibration_performed=False, audio_mux_match_checked=False,
                  evaluation_scope='Complete selected video/audio decode and input evidence, plus unaligned RGB differences. '
                      'Not continuous audio listening, pose ground truth, dance/music accuracy or final model acceptance. '
                      'Predicted token length changes packed layout and remains a potential causal confound.')
    # Group markers are not publishable review evidence until the final report exists.
    for path, image in pending_mosaics:
        with path.open('xb') as stream:
            image.save(stream, format='PNG')
    for path, report in pending_groups:
        atomic_write_new(path, report)
    atomic_write_new(out, result)
    print(json.dumps(dict(event='diagnostic_audit_finished', passed=9, pixel_pairs=pixel_pairs, out=str(out))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['plan', 'audit'], required=True)
    parser.add_argument('--exp_root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', nargs=2)
    args = parser.parse_args()
    if args.out.exists() or not args.out.parent.is_dir():
        parser.error('Require a new report and existing parent')
    if args.stage == 'plan':
        if not args.jobs:
            parser.error('Plan requires the two real managed job handles')
        diagnostic_plan(args.exp_root, args.out, args.jobs)
    else:
        audit(args.exp_root, args.out)


if __name__ == '__main__':
    main()
