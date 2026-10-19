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


def online_audit_main():
    import decord
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    parser = argparse.ArgumentParser(description='Online A/B training: actual input, full video/audio and counterfactual audit')
    parser.add_argument('--online', action='store_true', required=True)
    parser.add_argument('--exp_root', type=Path, required=True)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--tags', nargs='+', required=True)
    parser.add_argument('--zero_tags', nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if (args.out.exists() or not args.out.parent.is_dir()
            or len(args.runs) != len(args.tags) or len(args.runs) != len(args.zero_tags)
            or len(set(args.tags + args.zero_tags)) != 2 * len(args.runs)):
        parser.error('Require new report, existing parent and matching unique video groups')
    if any(Path(name).name != name or name in ('.', '..') for name in args.runs + args.tags + args.zero_tags):
        parser.error('Run and group names must be path-free identifiers')
    exp = args.exp_root
    cases = fixed_cases(exp)
    rows, mosaics, pending_images, pending_markers, models = [], [], [], [], []
    for run, tag, zero_tag in zip(args.runs, args.tags, args.zero_tags):
        checkpoint = exp / 'MS' / run / 'last.pt'
        summary = json.loads((checkpoint.parent / 'finished.json').read_text())
        ckpt_sha = sha256(checkpoint)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        metadata = state['metadata']
        train_backbone = summary['protocol']['backbone_train']
        if type(train_backbone) is not bool:
            raise ValueError('Training scope must be explicit')
        if (not summary['complete'] or summary['checkpoint_sha256'] != ckpt_sha
                or not summary['deployment']['independent_inference_tail_exact']
                or metadata['format_version'] != 2 or metadata['online_training']['feature_cache'] != 'forbidden'
                or metadata['online_training']['train_backbone'] != train_backbone
                or metadata['freeze_backbone'] != (not train_backbone)):
            raise ValueError('Require completed online models with exact full deployment and consistent scope')
        qmax = metadata['query_max']
        del state
        models.append(dict(run=run, checkpoint_sha256=ckpt_sha,
                           backbone_train=train_backbone, steps=summary['steps'], final=summary['final']))
        for group_tag, conditions, zero in ((tag, ('joint', 'aswap'), False), (zero_tag, ('joint',), True)):
            directory = exp / 'H3AR' / ('val_musicreplace_' + group_tag)
            jobs = json.loads((directory / 'jobs.json').read_text())
            if not jobs['all_sampler_jobs_exit_zero']:
                raise ValueError('Incomplete online generation jobs')
            group_rows = []
            for condition in conditions:
                grid = Image.new('RGB', (176 * 7, 348 * 3), (245, 245, 245))
                draw = ImageDraw.Draw(grid)
                for i, case in enumerate(c for c in cases if c['condition'] == condition):
                    evidence = successful_log(directory / 'logs', case, require_replacement=False, require_empty_prompt=True)
                    if evidence is None:
                        raise ValueError('Missing successful empty-student generation evidence')
                    log_path, log_sha, log_signature, _ = evidence
                    text = log_path.read_text()
                    inputs = one_record(text, '[h3ar] input_conditions ')
                    deployment = one_record(text, '[music text deployment] ')
                    native = one_record(text, '[music text native isolation] ')
                    if (inputs.get('music_text_ckpt') != str(checkpoint)
                            or inputs.get('music_text_zero_input') != zero
                            or inputs.get('text_hidden_oracle') is not False
                            or inputs.get('legacy_clip_complete_chain') is not False
                            or inputs.get('native_audio_condition') != 'fixed_silence'
                            or inputs.get('waveform_loader') != 'music_clip_bridge.load_audio'):
                        raise ValueError('Online trained model was not deployed with the declared input route')
                    shape = deployment.get('tail_shape', [])
                    if (deployment.get('checkpoint_sha256') != ckpt_sha or len(shape) != 2 or shape[1] != 5120
                            or not 1 <= shape[0] <= qmax or shape[0] != deployment.get('predicted_length')
                            or deployment.get('uses_ground_truth_caption') is not False
                            or deployment.get('final_adapter_projection_executed') is not True
                            or deployment.get('legacy_clip_complete_chain') is not False
                            or native.get('native_music_bypass') is not False
                            or native.get('all_target_audio_clean') is not True):
                        raise ValueError('Online raw H mapping or native isolation failed')
                    video = directory / condition / case['filename']
                    before = signature(video)
                    integrity = probe_video(video)
                    reader = decord.VideoReader(str(video), num_threads=1)
                    with Image.open(inputs['image_path']) as image:
                        reference = image.convert('RGB')
                    images = [reference] + [Image.fromarray(reader[f].asnumpy()) for f in (0, 24, 48, 72, 96, 123)]
                    for col, image in enumerate(images):
                        grid.paste(image.resize((176, 320)), (col * 176, i * 348 + 28))
                        draw.text((col * 176 + 4, i * 348 + 8), case['sample_id'] if col == 0 else str((0, 24, 48, 72, 96, 123)[col - 1] / 24) + 's', fill='black')
                    row = dict(tag=group_tag, condition=condition, sample_id=case['sample_id'],
                        video_path=str(video), video_sha256=sha256(video), audio_path=case['audio_path'],
                        log_path=str(log_path), log_sha256=log_sha, inputs=inputs, deployment=deployment,
                        native_audio=native, integrity=integrity, backbone_train=train_backbone, zero_input=zero,
                        audio_mux_match_checked=False, video_quality_acceptance_checked=False)
                    if signature(video) != before or signature(log_path) != log_signature:
                        raise ValueError('Online video or matching log changed during audit')
                    rows.append(row)
                    group_rows.append(row)
                mosaic = directory / (args.out.stem + '_' + condition + '_mosaic.png')
                if mosaic.exists():
                    raise FileExistsError(mosaic)
                pending_images.append((mosaic, grid))
                mosaics.append(dict(tag=group_tag, condition=condition, path=str(mosaic)))
            marker = directory / ('all6_integrity.json' if len(conditions) == 2 else 'joint_integrity.json')
            if marker.exists():
                raise FileExistsError('Refuse existing online group acceptance marker')
            pending_markers.append((marker, dict(complete=True, expected=len(group_rows), passed=len(group_rows),
                rows=group_rows, final_online_report=str(args.out), musical_quality_accepted=False)))
            print(json.dumps(dict(event='online_group_decoded', tag=group_tag, videos=len(group_rows))), flush=True)
    expected_count = 9 * len(args.runs)
    if len(rows) != expected_count or any(len({r['native_audio'].get(key) for r in rows}) != 1 or
                              any(not r['native_audio'].get(key) for r in rows)
                              for key in ('waveform_sha256', 'reference_sha256', 'retake_sha256', 'mask_sha256')):
        raise ValueError('All clips must have identical native silence conditions')
    lookup = {(r['tag'], r['condition'], r['sample_id']): r for r in rows}
    pixel_pairs = []
    for tag, zero_tag in zip(args.tags, args.zero_tags):
        for case in cases[:3]:
            sid = case['sample_id']
            selected = [lookup[tag, 'joint', sid], lookup[tag, 'aswap', sid], lookup[zero_tag, 'joint', sid]]
            readers = [decord.VideoReader(r['video_path'], num_threads=1) for r in selected]
            differences, movement, previous = [[], []], [[], [], []], [None, None, None]
            for frame in range(124):
                values = [reader[frame].asnumpy().astype(np.float32) / 255 for reader in readers]
                for j in (0, 1):
                    differences[j].append(float(np.abs(values[0] - values[j + 1]).mean()))
                for j, value in enumerate(values):
                    if previous[j] is not None:
                        movement[j].append(float(np.abs(value - previous[j]).mean()))
                    previous[j] = value
            pixel_pairs.append(dict(tag=tag, sample_id=sid,
                original_swapped_rgb_mae=float(np.mean(differences[0])), original_zero_rgb_mae=float(np.mean(differences[1])),
                adjacent_frame_rgb_mae=[float(np.mean(value)) for value in movement],
                predicted_lengths=[r['deployment']['predicted_length'] for r in selected]))
    result = dict(complete=True, expected=expected_count, passed=expected_count, models=models, rows=rows, mosaics=mosaics,
        pixel_pairs=pixel_pairs, all_native_audio_hashes_identical=True,
        student_ground_truth_caption_or_length_input=False, complete_selected_audio_video_decode=True,
        audio_mux_match_checked=False, musical_quality_accepted=False,
        scope='Technical integrity and unaligned pixel proxies only; not pose/music accuracy or continuous listening. Predicted length remains a packed-layout confound.')
    for path, image in pending_images:
        with path.open('xb') as stream:
            image.save(stream, format='PNG')
    atomic_write_new(args.out, result)
    for path, record in pending_markers:
        atomic_write_new(path, record)
    print(json.dumps(dict(event='online_all_audited', passed=expected_count, out=str(args.out), pixel_pairs=pixel_pairs)), flush=True)


def common_interval_main():
    import numpy as np
    import torch
    from music_text_adapter import load_music_text_checkpoint
    from music_text_contrastive import (
        _online_batch, _online_conditions, _online_scores, _state_hashes, score_samples,
    )
    parser = argparse.ArgumentParser(description='Read-only scoring-length ablation of completed online checkpoints')
    parser.add_argument('--common_interval', action='store_true', required=True)
    parser.add_argument('--exp_root', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if (args.out.exists() or not args.out.parent.is_dir() or len(set(args.runs)) != len(args.runs)
            or any(Path(name).name != name or name in ('.', '..') for name in args.runs)):
        parser.error('Require unique path-free run names and a new report with an existing parent')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    bundle_hash = sha256(args.bundle)
    bundle = torch.load(args.bundle, map_location='cpu', weights_only=True)
    mean, std = bundle['mean'].cuda(), bundle['std'].cuda()
    models = []
    for run in args.runs:
        directory = args.exp_root / 'MS' / run
        summary = json.loads((directory / 'finished.json').read_text())
        checkpoint = directory / 'last.pt'
        if (summary.get('complete') is not True or sha256(checkpoint) != summary['checkpoint_sha256']
                or bundle_hash != summary['protocol']['online_bundle_sha256']):
            raise ValueError('Completed checkpoint or waveform bundle identity differs from its training record')
        ids = summary['protocol']['validation_ids']
        saved = json.loads((directory / 'final_validation.json').read_text())
        if ids != bundle['metadata']['val_ids'] or saved['candidates'] != len(ids):
            raise ValueError('Validation pool differs from the completed experiment')
        lookup = {row['sample_id']: row for row in saved['cases']}
        if len(lookup) != len(ids) or set(lookup) != set(ids):
            raise ValueError('Saved validation rows are missing or duplicated')
        model = load_music_text_checkpoint(checkpoint, expected_backend='musicfm').cuda().eval()
        before = _state_hashes(model)
        if before != json.loads((directory / 'weight_update_audit.json').read_text())['after_sha256']:
            raise ValueError('Loaded parameters or buffers differ from the completed weight audit')
        rows, maximum_error = [], 0.
        with torch.no_grad():
            for start in range(0, len(ids), summary['protocol']['microbatch']):
                names = ids[start:start + summary['protocol']['microbatch']]
                wave, context, target, lengths, teacher_mask = _online_batch(names, bundle, model.adapter.qmax)
                conditions, wrong_ids, second_ids = _online_conditions(model, names, ids, bundle, context, 1)
                zero = model.student_condition(torch.zeros_like(wave), context)
                outputs = conditions + [zero]
                native = _online_scores(outputs, target, teacher_mask, mean, std)
                predicted = torch.stack([output['lengths'] for output in outputs], dim=1)
                common_lengths = predicted.min(dim=1).values
                common_mask = torch.arange(model.adapter.qmax, device='cuda')[None] < common_lengths[:, None]
                common = torch.stack([score_samples(output['raw_tokens'], target, teacher_mask, mean, std,
                                                     student_mask=common_mask) for output in outputs], dim=1)
                for i, sid in enumerate(names):
                    old = lookup[sid]
                    if (wrong_ids[i] != old['wrong_audio_id'] or second_ids[i] != old['second_audio_id']
                            or predicted[i].tolist() != old['predicted_lengths'] + [old['zero_length']]):
                        raise ValueError('Original counterfactual IDs or predicted lengths were not reproduced')
                    error = max(abs(a - b) for a, b in zip(native[i].tolist(), old['scores'] + [old['zero_score']]))
                    maximum_error = max(maximum_error, error)
                    if error > 1e-6:
                        raise ValueError('Original scores were not reproduced: ' + sid)
                    overlap = min(int(common_lengths[i]), int(lengths[i]))
                    rows.append(dict(sample_id=sid, wrong_audio_id=wrong_ids[i], second_audio_id=second_ids[i],
                        target_length_supervision_only=int(lengths[i]), predicted_lengths_original=predicted[i].tolist(),
                        common_score_interval=int(common_lengths[i]), native_scores_reproduced=native[i].tolist(),
                        common_interval_scores=common[i].tolist(),
                        common_interval_margin_wrong=float(common[i, 0] - common[i, 1]),
                        common_interval_margin_second=float(common[i, 0] - common[i, 2]),
                        common_interval_margin_zero=float(common[i, 0] - common[i, 3]),
                        shared_teacher_overlap_mse=[float((output['raw_tokens'][i, :overlap] - target[i, :overlap]).square().mean())
                                                    for output in outputs], overlap_length_supervision_only=overlap))
                del conditions, outputs, zero, native, common, wave, context, target
        if before != _state_hashes(model):
            raise ValueError('Read-only evaluation changed model parameters or buffers')
        native_margin = [row['native_scores_reproduced'][0] - row['native_scores_reproduced'][1] for row in rows]
        closeness = [abs(row['predicted_lengths_original'][1] - row['target_length_supervision_only'])
                     - abs(row['predicted_lengths_original'][0] - row['target_length_supervision_only']) for row in rows]
        correlation = float(np.corrcoef(native_margin, closeness)[0, 1]) if np.std(native_margin) > 0 and np.std(closeness) > 0 else None
        stats = dict(native_margin_wrong_mean=float(np.mean(native_margin)),
                     native_margin_vs_target_length_closeness_pearson=correlation,
                     common_score_interval_mean=float(np.mean([row['common_score_interval'] for row in rows])),
                     common_teacher_overlap_correct_better_fraction=float(np.mean([
                         row['shared_teacher_overlap_mse'][1] - row['shared_teacher_overlap_mse'][0] > 1e-6 for row in rows])))
        for key in ('wrong', 'second', 'zero'):
            margins = np.asarray([row['common_interval_margin_' + key] for row in rows])
            stats['common_interval_margin_' + key + '_mean'] = float(margins.mean())
            stats['common_interval_correct_beats_' + key + '_fraction'] = float((margins > 1e-6).mean())
        models.append(dict(run=run, steps=summary['steps'], checkpoint_sha256=summary['checkpoint_sha256'],
                           candidates=len(rows), native_scores_max_abs_error=maximum_error,
                           original_lengths_exact=True, parameters_and_buffers_unchanged=True,
                           statistics=stats, cases=rows))
        print(json.dumps(dict(event='common_interval_model_finished', run=run, statistics=stats)), flush=True)
        del model
        torch.cuda.empty_cache()
    atomic_write_new(args.out, dict(complete=True, models=models, bundle_sha256=bundle_hash, optimizer_updates=0,
        student_generation_condition_changed=False, student_reads_caption_or_teacher_length=False,
        evaluation_only_student_mask='Same per-case min of correct/two_wrong/zero predicted lengths; common visible query intersection',
        teacher_interval='GT mask remains supervision only', native_output_and_score_reproduction_verified=True,
        scope='Read-only representation diagnostic, not equal-layout video intervention, new training or causal proof'))


def main():
    if '--common_interval' in __import__('sys').argv:
        return common_interval_main()
    if '--online' in __import__('sys').argv:
        return online_audit_main()
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
