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
    parser.add_argument('--no_text_encoder', action='store_true', help='Audit V3 music-only models with no blank text, TC prefix or external context')
    parser.add_argument('--checkpoint_cache_root', type=Path, help='Verified local byte-for-byte checkpoint cache; canonical training summaries remain authoritative')
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
        canonical_checkpoint = exp / 'MS' / run / 'last.pt'
        summary = json.loads((canonical_checkpoint.parent / 'finished.json').read_text())
        if args.checkpoint_cache_root and not args.no_text_encoder:
            parser.error('Local checkpoint cache is only enabled for the explicit noTC audit')
        checkpoint = args.checkpoint_cache_root / run / 'last.pt' if args.checkpoint_cache_root else canonical_checkpoint
        ckpt_sha = sha256(checkpoint)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        metadata = state['metadata']
        train_backbone = summary['protocol']['backbone_train']
        if type(train_backbone) is not bool:
            raise ValueError('Training scope must be explicit')
        if (not summary['complete'] or summary['checkpoint_sha256'] != ckpt_sha
                or not summary['deployment']['independent_inference_tail_exact']
                or metadata['format_version'] != (3 if args.no_text_encoder else 2)
                or metadata['online_training']['feature_cache'] != 'forbidden'
                or metadata['online_training']['train_backbone'] != train_backbone
                or metadata['freeze_backbone'] != (not train_backbone)):
            raise ValueError('Require completed online models with exact full deployment and consistent scope')
        if args.no_text_encoder and (metadata.get('context_policy') != 'music_only_global_bias'
                or summary['protocol'].get('no_text_encoder') is not True
                or summary['deployment'].get('cached_context_read') is not False
                or 'adapter.global_context_bias' not in state['model']
                or any('context_projection' in key for key in state['model'])):
            raise ValueError('No-TC audit requires a trained waveform-only global-bias model, not a legacy zero-context diagnostic')
        qmax = metadata['query_max']
        del state
        models.append(dict(run=run, checkpoint_sha256=ckpt_sha,
                           checkpoint_source=str(canonical_checkpoint), checkpoint_inference_path=str(checkpoint),
                           backbone_train=train_backbone, steps=summary['steps'], final=summary['final']))
        for group_tag, conditions, zero in ((tag, ('joint', 'aswap'), False), (zero_tag, ('joint',), True)):
            directory = exp / 'H3AR' / ('val_musicreplace_' + group_tag)
            jobs = json.loads((directory / 'jobs.json').read_text())
            if not jobs['all_sampler_jobs_exit_zero']:
                raise ValueError('Incomplete online generation jobs')
            if args.no_text_encoder:
                plan = json.loads((directory / 'plan.json').read_text())
                if (plan.get('checkpoint_sha256') != ckpt_sha or plan.get('checkpoint_source') != str(canonical_checkpoint)
                        or plan.get('checkpoint_inference_path') != str(checkpoint)):
                    raise ValueError('No-TC inference cache is not bound to its canonical completed training checkpoint')
            group_rows = []
            for condition in conditions:
                grid = Image.new('RGB', (176 * 7, 348 * 3), (245, 245, 245))
                draw = ImageDraw.Draw(grid)
                for i, case in enumerate(c for c in cases if c['condition'] == condition):
                    evidence = successful_log(directory / 'logs', case, require_replacement=False, require_empty_prompt=not args.no_text_encoder)
                    if evidence is None:
                        raise ValueError('Missing successful shared-student generation evidence')
                    log_path, log_sha, log_signature, _ = evidence
                    text = log_path.read_text()
                    inputs = one_record(text, '[h3ar] input_conditions ')
                    deployment = one_record(text, '[music text deployment] ')
                    native = one_record(text, '[no tc native isolation] ' if args.no_text_encoder else '[music text native isolation] ')
                    if args.no_text_encoder:
                        condition_record = one_record(text, '[no tc condition] ')
                        if (inputs.get('semantic_text', 'missing') is not None
                                or inputs.get('no_text_encoder') is not True or inputs.get('text_encoder_loaded') is not False
                                or condition_record.get('kind') != 'music_only_global_bias_no_tc_deployment'
                                or condition_record.get('old_model_not_trained_for_no_tc') is not False
                                or condition_record.get('context_source') != 'none_music_only_global_bias'
                                or condition_record.get('loaded') is not False
                                or not no_tc_count(condition_record.get('calls'), 0)
                                or not no_tc_count(condition_record.get('prefix_length'), 0)
                                or condition_record.get('semantic_text', 'missing') is not None
                                or condition_record.get('processor_loaded') is not False or condition_record.get('tokenizer_loaded') is not False
                                or condition_record.get('checkpoint_sha256') != ckpt_sha
                                or condition_record.get('tail_shape') != deployment.get('tail_shape')
                                or condition_record.get('predicted_length') != deployment.get('predicted_length')
                                or native.get('text_encoder_loaded') is not False or native.get('tokenizer_loaded') is not False
                                or native.get('processor_loaded') is not False or native.get('semantic_text', 'missing') is not None
                                or not no_tc_count(native.get('text_encoder_calls'), 0)
                                or not no_tc_count(native.get('prefix_tokens'), 0)
                                or native.get('prompt_embeds_shape') != deployment.get('tail_shape')
                                or type(native.get('visual_reference_rows')) is not int or native['visual_reference_rows'] <= 0
                                or not no_tc_hash(native.get('visual_reference_sha256'))):
                            raise ValueError('Trained no-TC deployment has text/context/prefix contamination or changed raw-token boundary')
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
                        no_text_encoder=args.no_text_encoder,
                        no_tc_condition=condition_record if args.no_text_encoder else None,
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
    if args.no_text_encoder:
        for sid in {case['sample_id'] for case in cases}:
            selected = [row for row in rows if row['sample_id'] == sid]
            if any(len({row['native_audio'][key] for row in selected}) != 1
                   for key in ('visual_reference_rows', 'visual_reference_sha256')):
                raise ValueError('No-TC original/swapped/zero or model groups changed the same-image VAE reference')
    result = dict(complete=True, expected=expected_count, passed=expected_count, models=models, rows=rows, mosaics=mosaics,
        no_text_encoder=args.no_text_encoder,
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


def projection_basis(matrix):
    import torch
    if matrix.ndim != 2 or not torch.isfinite(matrix).all():
        raise ValueError('Projection requires a finite matrix')
    u, values, _ = torch.linalg.svd(matrix.double(), full_matrices=False)
    tolerance = max(matrix.shape) * torch.finfo(torch.float64).eps * float(values[0])
    rank = int((values > tolerance).sum())
    if rank < 1:
        raise ValueError('Projection space has zero numerical rank')
    basis = u[:, :rank]
    error = float((basis.T @ basis - torch.eye(rank, device=basis.device, dtype=basis.dtype)).abs().max())
    if error > 1e-8:
        raise ValueError('Projection basis is not orthonormal')
    return basis, dict(numerical_rank=rank, svd_dtype='float64', rank_tolerance=tolerance,
                       basis_orthogonality_max_abs=error, singular_values=values.tolist())


def project_raw_tokens(tail, center, basis):
    import torch
    value = tail.double() - center.double()
    projected = center.double() + (value @ basis.double()) @ basis.double().T
    if projected.shape != tail.shape or not torch.isfinite(projected).all():
        raise ValueError('Invalid projected raw condition')
    return projected


def nested_pca_bases(centered):
    import torch
    if (centered.ndim != 2 or min(centered.shape) < 544
            or centered.dtype != torch.float32 or not torch.isfinite(centered).all()):
        raise ValueError('Nested PCA requires a finite FP32 weighted matrix with dimensions >= 544')
    devices = [centered.device.index] if centered.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(42)
        _, singular, directions = torch.pca_lowrank(centered, q=544, center=False, niter=4)
    basis512, _ = torch.linalg.qr(directions[:, :512].double(), mode='reduced')
    basis256 = basis512[:, :256]
    orthogonality = float((basis512.T @ basis512 - torch.eye(
        512, dtype=basis512.dtype, device=basis512.device)).abs().max())
    nesting = float((basis512 @ (basis512.T @ basis256) - basis256).abs().max())
    if orthogonality > 1e-8 or nesting > 1e-8:
        raise ValueError('Nested PCA basis failed orthogonality or containment')
    energy = centered.double().square().sum()
    if float(energy) <= 0:
        raise ValueError('Nested PCA training matrix has zero energy')
    energies = {rank: float((centered.double() @ basis512[:, :rank]).square().sum() / energy)
                for rank in (256, 512)}
    metadata = dict(algorithm='torch.pca_lowrank_randomized', oversampled_rank=544,
        power_iterations=4, seed=42, compute_dtype='float32', final_basis_dtype='float64',
        qr_performed_once_rank=512,
        basis512_orthogonality_max_abs=orthogonality, nesting_residual_max_abs=nesting,
        basis256_exact_prefix_of_basis512=True, retained_sample_energy_fraction_rank512=energies[512],
        retained_sample_energy_fraction_by_rank={str(rank): value for rank, value in energies.items()},
        singular_values=singular.tolist(), exact_optimal_pca_claim=False)
    return basis256, basis512, metadata


def projection_main():
    import hashlib
    import torch
    from music_text_behavior import save_new
    parser = argparse.ArgumentParser(description='No-update oracle projection into current output spaces and train-only PCA')
    parser.add_argument('--projection', action='store_true', required=True)
    parser.add_argument('--exp_root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--oracle_cache', type=Path, required=True)
    parser.add_argument('--pca_nested', action='store_true',
                        help='Only nested FP32 train-PCA256/512 from one q544 fit with final FP64 QR; no W/LN variants')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not args.out.parent.is_dir():
        parser.error('Require new projection directory and an existing parent')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    summary = json.loads((args.checkpoint.parent / 'finished.json').read_text())
    checkpoint_sha = sha256(args.checkpoint)
    bundle_sha = sha256(args.bundle)
    if (summary.get('complete') is not True or checkpoint_sha != summary['checkpoint_sha256']
            or summary['protocol']['online_bundle_sha256'] != bundle_sha):
        raise ValueError('Projection sources differ from the completed training identity')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['metadata'].get('format_version') != 2:
        raise ValueError('Require an audited complete online checkpoint')
    state = checkpoint['model']
    variants = {}
    if not args.pca_nested:
        w = state['adapter.raw_output.weight'].cuda().double()
        b = state['adapter.raw_output.bias'].cuda().double()
        gamma = state['adapter.output_norm.weight'].cuda().double()
        beta = state['adapter.output_norm.bias'].cuda().double()
        if w.shape != (5120, 256) or b.shape != (5120,) or gamma.shape != (256,):
            raise ValueError('Unexpected fixed adapter output structure')
        w_basis, w_meta = projection_basis(w)
        zero_sum = torch.zeros(256, 255, dtype=torch.float64, device='cuda')
        zero_sum[:255] = torch.eye(255, device='cuda', dtype=torch.float64)
        zero_sum[-1] = -1
        zero_sum, _ = torch.linalg.qr(zero_sum, mode='reduced')
        ln_basis, ln_meta = projection_basis((w * gamma[None]) @ zero_sum)
        ln_center = b + w @ beta
        variants.update(w256=(b, w_basis, w_meta), ln255=(ln_center, ln_basis, ln_meta))
    bundle = torch.load(args.bundle, map_location='cpu', weights_only=True)
    train = bundle['metadata']['train_ids']
    if train != summary['protocol']['train_ids'] or len(train) != len(set(train)):
        raise ValueError('PCA train pool is not the declared original train pool')
    samples, sampling, sample_weights = [], [], []
    train_sum = torch.zeros(5120, dtype=torch.float64)
    train_token_count = 0
    for sid in train:
        tail = bundle['targets'][sid]['tail']
        train_sum += tail.double().sum(0)
        train_token_count += len(tail)
        indices = torch.linspace(0, len(tail) - 1, min(16, len(tail))).round().long()
        samples.append(tail[indices].float())
        sample_weights.append(torch.full((len(indices),), (len(tail) / len(indices)) ** 0.5))
        sampling.append(dict(sample_id=sid, original_tokens=len(tail), selected_indices=indices.tolist()))
    matrix = torch.cat(samples).cuda()
    center = (train_sum / train_token_count).cuda()
    centered = (matrix - center.float()) * torch.cat(sample_weights).cuda()[:, None]
    sampling_meta = dict(train_samples=len(train), sampled_tokens=len(matrix), max_tokens_per_train_item=16,
        center_source='Exact FP64 mean of all valid training tail tokens', full_train_tokens=train_token_count,
        covariance_sampling='Deterministic evenly-spaced tokens, weighted by original item length/selected count; approximate',
        sampling=sampling, prefix_and_padding_used=False, validation_used=False, exact_optimal_pca_claim=False)
    nesting_metadata = None
    if args.pca_nested:
        basis256, basis512, nesting_metadata = nested_pca_bases(centered)
        for rank, basis in ((256, basis256), (512, basis512)):
            metadata = dict(sampling_meta, **nesting_metadata, rank=rank,
                retained_sample_energy_fraction=nesting_metadata['retained_sample_energy_fraction_by_rank'][str(rank)])
            variants['pca' + str(rank) + 'n'] = (center, basis, metadata)
    else:
        _, singular, directions = torch.pca_lowrank(centered, q=288, center=False, niter=4)
        pca_basis = directions[:, :256].double()
        pca_basis, _ = torch.linalg.qr(pca_basis, mode='reduced')
        pca_meta = dict(sampling_meta, algorithm='torch.pca_lowrank_randomized', rank=256, oversampled_rank=288,
            power_iterations=4, seed=42, compute_dtype='float32', retained_sample_energy_fraction=float(
                ((centered.double() @ pca_basis).square().sum() / centered.double().square().sum())))
        variants['pca256'] = (center, pca_basis, pca_meta)
    del matrix, centered, checkpoint
    cases = [case for case in fixed_cases(args.exp_root) if case['condition'] == 'joint']
    if len(cases) != 3 or len({case['sample_id'] for case in cases}) != 3:
        raise ValueError('Projection requires exactly three unique fixed joint cases')
    oracle_records = []
    for case in cases:
        source = args.oracle_cache / (Path(case['filename']).stem + '.pt')
        record = torch.load(source, map_location='cpu', weights_only=True)
        meta = record['metadata']
        if (meta['kind'] != 'oracle_teacher_hidden_not_music_prediction'
                or meta['case']['sample_id'] != case['sample_id'] or meta['case']['audio_path'] != case['audio_path']
                or meta['case']['condition'] != 'joint' or meta['tail_length'] != len(record['tail'])):
            raise ValueError('Require the existing neutral same-reference joint oracle')
        image_path = Path(meta['case']['image_path'])
        oracle_records.append((case, source, record, sha256(image_path)))
    args.out.mkdir()
    manifests, reports = [], []
    for name, (projection_center, basis, metadata) in variants.items():
        entries = {}
        variant_rows = []
        numeric = args.out / (name + '_space.pt')
        save_new(numeric, dict(center=projection_center.cpu(), basis=basis.cpu()), binary=True)
        for case, source, record, image_sha in oracle_records:
            original = record['tail'].cuda().double()
            projected = project_raw_tokens(original, projection_center, basis)
            repeated = project_raw_tokens(projected, projection_center, basis)
            if float((repeated - projected).abs().max()) > 1e-6:
                raise ValueError('Affine projection is not idempotent')
            deployed = projected.to(torch.bfloat16).cpu()
            metrics = dict(raw_mse=float((original - projected).square().mean()),
                raw_relative_rms=float((original - projected).norm() / original.norm().clamp_min(1e-12)),
                bf16_deployed_mse=float((original.cpu() - deployed.double()).square().mean()),
                bf16_quantization_mse=float((projected.cpu() - deployed.double()).square().mean()),
                projection_idempotence_max_abs=float((repeated - projected).abs().max()),
                teacher_tokens=len(original), deployed_tokens=len(deployed))
            path = args.out / (name + '_' + case['sample_id'] + '.pt')
            payload = dict(tail=deployed, original_tail=record['tail'], prefix_context=record['prefix_context'],
                prefix_tags=record['prefix_tags'], prefix_ids=record['token_ids'][:len(record['prefix_tags'])],
                sample_id=case['sample_id'], variant=name, oracle_file_sha256=sha256(source), metrics=metrics)
            save_new(path, payload, binary=True)
            entries[case['sample_id']] = dict(path=str(path), file_sha256=sha256(path),
                image_path=record['metadata']['case']['image_path'], image_sha256=image_sha,
                audio_path=case['audio_path'], oracle_source=str(source), oracle_sha256=sha256(source),
                projected_tail_sha256=hashlib.sha256(deployed.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                original_teacher_tail_sha256=record['metadata']['teacher_tail_sha256'],
                teacher_tokens=len(original), variant=name)
            variant_rows.append(dict(sample_id=case['sample_id'], metrics=metrics))
        validation = []
        for sid in bundle['metadata']['val_ids']:
            original = bundle['targets'][sid]['tail'].cuda().double()
            projected = project_raw_tokens(original, projection_center, basis)
            validation.append(dict(sample_id=sid, tokens=len(original), raw_mse=float((original - projected).square().mean())))
        manifest = args.out / (name + '_manifest.json')
        atomic_write_new(manifest, dict(complete=True, format_version=1, diagnostic_only=True,
            kind='projected_teacher_not_music_prediction', variant=name, cases=entries,
            checkpoint_sha256=checkpoint_sha, bundle_sha256=bundle_sha,
            train_ids=train, space_metadata=metadata,
            teacher_length_for_oracle_only=True, changes_student_deployment=False,
            output_space=str(numeric), output_space_sha256=sha256(numeric)))
        manifests.append(str(manifest))
        reports.append(dict(variant=name, space_metadata=metadata, generation_cases=variant_rows,
            validation_projection_rows=validation, validation_projection_macro_mse=sum(r['raw_mse'] for r in validation) / len(validation),
            oracle_manifest=str(manifest), interpretation='Relaxed affine-envelope or sampled train-PCA projection; not reachable student outputs or generation gain'))
    if args.pca_nested:
        residual_rows = []
        for low, high in zip(reports[0]['validation_projection_rows'], reports[1]['validation_projection_rows']):
            if (low['sample_id'] != high['sample_id'] or low['tokens'] != high['tokens']
                    or high['raw_mse'] > low['raw_mse'] + 1e-10 * max(1., low['raw_mse'])):
                raise ValueError('Nested rank512 validation residual exceeds rank256')
            residual_rows.append(dict(sample_id=low['sample_id'], tokens=low['tokens'],
                raw_mse_rank256=low['raw_mse'], raw_mse_rank512=high['raw_mse']))
        nesting_metadata = dict(nesting_metadata, validation_residuals=residual_rows,
                                validation_rank512_residual_le_rank256=True)
    report = dict(complete=True, optimizer_updates=0, checkpoint_sha256=checkpoint_sha,
        checkpoint_summary_sha256=sha256(args.checkpoint.parent / 'finished.json'),
        diagnostic_source_sha256=sha256(Path(__file__)),
        bundle_sha256=bundle_sha, source_people=3, variants=reports, manifests=manifests,
        teacher_baseline_reused=str(args.oracle_cache.parent), pca_uses_train_only=True,
        scope='Numerical projection and same-length teacher-oracle inputs only; video decoding and human dance-quality review are separate')
    if args.pca_nested:
        report.update(pca_nested=True, nested_pca=nesting_metadata)
    atomic_write_new(args.out / 'finished.json', report)
    print(json.dumps(dict(event='teacher_projection_inputs_finished', manifests=manifests,
                         generation_video_jobs_expected=3 * len(variants), optimizer_updates=0)), flush=True)


def projection_audit_specs(exp_root, projection_dir, inputs, generation_dirs):
    baseline = exp_root / 'H3AR/val_musicreplace_text_oracle_neutral_v1'
    reports = inputs['variants']
    names = [report['variant'] for report in reports]
    if (not names or len(names) != len(set(names))
            or any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', name) for name in names)):
        raise ValueError('Require unique path-free finished projection variants')
    if generation_dirs is None:
        if set(names) != {'w256', 'ln255', 'pca256'}:
            raise ValueError('Nonhistorical projection variants require explicit --generation_dirs VARIANT=PATH')
        bindings = {name: exp_root / 'H3AR' / ('val_musicreplace_text_projection_C300_' + name + '_v2')
                    for name in ('w256', 'ln255', 'pca256')}
        names = ['w256', 'ln255', 'pca256']
    else:
        bindings = {}
        for value in generation_dirs:
            name, separator, path = value.partition('=')
            if not separator or not path or name not in names or name in bindings:
                raise ValueError('Bind each finished variant exactly once with --generation_dirs VARIANT=PATH')
            bindings[name] = Path(path)
        if set(bindings) != set(names):
            raise ValueError('Generation directories must exactly cover finished projection variants')
    if len({path.resolve() for path in [baseline] + list(bindings.values())}) != len(names) + 1:
        raise ValueError('Baseline and variant generation directories must be distinct')
    manifests = {report['variant']: Path(report['oracle_manifest']) for report in reports}
    if (set(inputs['manifests']) != {str(path) for path in manifests.values()}
            or any(path.resolve() != (projection_dir / (name + '_manifest.json')).resolve()
                   for name, path in manifests.items())):
        raise ValueError('Finished variants must bind the exact projection-directory manifests')
    return [('neutral_teacher_reused', baseline, None)] + [
        (name, bindings[name], manifests[name]) for name in names]


def projection_audit_main():
    import decord
    import numpy as np
    from PIL import Image, ImageDraw
    parser = argparse.ArgumentParser(description='Complete same-length projected-teacher video audit with reused neutral teacher baseline')
    parser.add_argument('--projection_audit', action='store_true', required=True)
    parser.add_argument('--exp_root', type=Path, required=True)
    parser.add_argument('--projection_dir', type=Path, required=True)
    parser.add_argument('--generation_dirs', nargs='+', metavar='VARIANT=PATH',
                        help='Explicit generation directory for every finished variant; default is historical W/LN/PCA v2')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not args.out.parent.is_dir():
        parser.error('Require a new report and existing parent')
    inputs = json.loads((args.projection_dir / 'finished.json').read_text())
    if inputs.get('complete') is not True or inputs.get('optimizer_updates') != 0:
        raise ValueError('Projection inputs are incomplete or involved training')
    cases = [case for case in fixed_cases(args.exp_root) if case['condition'] == 'joint']
    if len(cases) != 3 or len({case['sample_id'] for case in cases}) != 3:
        raise ValueError('Projection audit requires exactly three unique fixed joint cases')
    rows, mosaics, pending_images, pending_markers = [], [], [], []
    specs = projection_audit_specs(args.exp_root, args.projection_dir, inputs, args.generation_dirs)
    variant_names = [variant for variant, _, manifest in specs if manifest is not None]
    expected_new, expected = 3 * len(variant_names), 3 * len(specs)
    for variant, directory, manifest_path in specs:
        jobs = json.loads((directory / 'jobs.json').read_text())
        if jobs.get('all_sampler_jobs_exit_zero') is not True or jobs['expected'] != 3:
            raise ValueError('Projection group has incomplete sampler jobs')
        manifest = json.loads(manifest_path.read_text()) if manifest_path else None
        if manifest is not None and (
                manifest.get('complete') is not True or manifest.get('variant') != variant
                or manifest.get('kind') != 'projected_teacher_not_music_prediction'
                or manifest.get('diagnostic_only') is not True
                or manifest.get('teacher_length_for_oracle_only') is not True
                or manifest.get('changes_student_deployment') is not False
                or manifest.get('checkpoint_sha256') != inputs['checkpoint_sha256']
                or manifest.get('bundle_sha256') != inputs['bundle_sha256']
                or set(manifest['cases']) != {case['sample_id'] for case in cases}
                or sha256(Path(manifest['output_space'])) != manifest['output_space_sha256']):
            raise ValueError('Projection manifest/source space differs from its finished variant')
        grid = Image.new('RGB', (176 * 7, 348 * 3), 'white')
        draw = ImageDraw.Draw(grid)
        group_rows = []
        for index, case in enumerate(cases):
            evidence = successful_log(directory / 'logs', case, require_replacement=False, require_empty_prompt=True)
            if evidence is None:
                raise ValueError('Missing matching successful projection/baseline generation log')
            log_path, log_sha, _, _ = evidence
            text = log_path.read_text()
            actual = one_record(text, '[h3ar] input_conditions ')
            projection = None
            if manifest_path:
                projection = one_record(text, '[music text projection] ')
                entry = manifest['cases'][case['sample_id']]
                if (actual.get('text_hidden_projection') != str(manifest_path) or actual.get('music_text_ckpt') is not None
                        or actual.get('text_hidden_oracle') is not False or projection.get('variant') != variant
                        or projection.get('manifest_sha256') != sha256(manifest_path)
                        or projection.get('prefix_ids_tags_context_exact') is not True
                        or projection.get('original_packed_exact') is not True
                        or projection.get('music_prediction') is not False
                        or projection.get('teacher_tokens') != entry['teacher_tokens']
                        or entry.get('variant') != variant
                        or actual.get('image_path') != entry['image_path']
                        or sha256(Path(entry['image_path'])) != entry['image_sha256']
                        or projection.get('projected_file_sha256') != entry['file_sha256']
                        or projection.get('projected_file_sha256') != sha256(Path(entry['path']))
                        or projection.get('oracle_source_sha256') != entry['oracle_sha256']
                        or projection.get('oracle_source_sha256') != sha256(Path(entry['oracle_source']))):
                    raise ValueError('Projected condition identity, source or same-layout evidence failed')
                native = projection['native_audio']
            else:
                records = jobs['condition_reports']
                native = next(record['native_audio'] for record in records
                              if record['case']['sample_id'] == case['sample_id'])
                if actual.get('text_hidden_oracle') is not True or actual.get('music_text_ckpt') is not None:
                    raise ValueError('Baseline is not the reused neutral teacher oracle')
            if (actual.get('native_audio_condition') != 'fixed_silence'
                    or native.get('native_music_bypass') is not False or native.get('all_target_audio_clean') is not True):
                raise ValueError('Real music bypass or incorrect native silence')
            video = directory / 'joint' / case['filename']
            before = signature(video)
            integrity = probe_video(video)
            reader = decord.VideoReader(str(video), num_threads=1)
            with Image.open(actual['image_path']) as image:
                reference = image.convert('RGB')
            images = [reference] + [Image.fromarray(reader[f].asnumpy()) for f in (0, 24, 48, 72, 96, 123)]
            for column, image in enumerate(images):
                grid.paste(image.resize((176, 320)), (176 * column, 348 * index + 28))
                draw.text((176 * column + 3, 348 * index + 6),
                          case['sample_id'] if column == 0 else str((0, 24, 48, 72, 96, 123)[column - 1] / 24) + 's', fill='black')
            row = dict(variant=variant, sample_id=case['sample_id'], condition='joint',
                video_path=str(video), video_sha256=sha256(video), audio_path=case['audio_path'],
                log_path=str(log_path), log_sha256=log_sha, inputs=actual, projection=projection,
                native_audio=native, integrity=integrity, baseline_reused=manifest_path is None,
                diagnostic_only=True, musical_quality_accepted=False)
            if signature(video) != before or sha256(log_path) != log_sha:
                raise ValueError('Projection video/log changed during audit')
            rows.append(row)
            group_rows.append(row)
        mosaic = directory / (args.out.stem + '_joint_mosaic.png')
        if mosaic.exists():
            raise FileExistsError(mosaic)
        pending_images.append((mosaic, grid))
        mosaics.append(dict(variant=variant, path=str(mosaic)))
        if manifest_path:
            marker = directory / 'projection_integrity.json'
            if marker.exists():
                raise FileExistsError(marker)
            pending_markers.append((marker, dict(complete=True, expected=3, passed=3, rows=group_rows,
                final_projection_report=str(args.out), diagnostic_only=True, teacher_length_only=True)))
        print(json.dumps(dict(event='projection_group_full_decoded', variant=variant, videos=3)), flush=True)
    if len(rows) != expected or any(len({row['native_audio'].get(key) for row in rows}) != 1
                                   or any(not row['native_audio'].get(key) for row in rows)
                                   for key in ('waveform_sha256', 'reference_sha256', 'retake_sha256', 'mask_sha256')):
        raise ValueError('All same-reference teacher/projection clips must have identical nonempty native silence hashes')
    pairs = []
    for sid in [case['sample_id'] for case in cases]:
        selected = [next(row for row in rows if row['sample_id'] == sid and row['variant'] == name)
                    for name in ['neutral_teacher_reused'] + variant_names]
        readers = [decord.VideoReader(row['video_path'], num_threads=1) for row in selected]
        differences = [[] for _ in variant_names]
        for frame in range(124):
            values = [reader[frame].asnumpy().astype(np.float32) / 255 for reader in readers]
            for index in range(len(variant_names)):
                differences[index].append(float(np.abs(values[0] - values[index + 1]).mean()))
        pairs.append(dict(sample_id=sid, teacher_vs_projection_rgb_mae={
            name: float(np.mean(value)) for name, value in zip(variant_names, differences)}))
    result = dict(complete=True, expected_new=expected_new, passed_new=expected_new, baseline_reused=3,
                  expected=expected, passed=expected,
                  generation_bindings=[dict(variant=name, directory=str(directory),
                      manifest=str(manifest) if manifest else None) for name, directory, manifest in specs],
                  projection_finished_sha256=sha256(args.projection_dir / 'finished.json'),
                  rows=rows, mosaics=mosaics, pixel_pairs=pairs, same_teacher_lengths_and_original_packed_verified=True,
                  all_native_silence_hashes_identical=True, optimizer_updates=0,
                  continuous_human_review='pending', musical_gain_accepted=False,
                  scope='Projected teacher-oracle space diagnostic, not learned music generation; RGB proxies are not dance-quality scores')
    for path, image in pending_images:
        with path.open('xb') as stream:
            image.save(stream, format='PNG')
    atomic_write_new(args.out, result)
    for path, record in pending_markers:
        atomic_write_new(path, record)


NO_TC_TEACHER_TAG = 'text_no_tc_teacher_v1'
NO_TC_STUDENT_TAG = 'text_no_tc_C300_v1'
NO_TC_TEACHER_KIND = 'cached_teacher_tail_only_no_tc_diagnostic'
NO_TC_STUDENT_KIND = 'legacy_checkpoint_no_tc_music_only_diagnostic'
NO_TC_C300_SHA = '189a0a74ea1789b9a4bc1b8d1f202264c354f418b0881339eca10baf0673f605'
NO_TC_NATIVE_KEYS = ('waveform_sha256', 'reference_sha256', 'retake_sha256', 'mask_sha256')


def no_tc_hash(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def no_tc_count(value, expected):
    return type(value) is int and value == expected


def no_tc_cases(exp):
    import csv
    cases = [case for case in fixed_cases(exp) if case['condition'] == 'joint']
    with (exp / 'E2/full_joint.csv').open(newline='', encoding='utf-8') as stream:
        originals = list(csv.DictReader(stream))[:3]
    if len(cases) != 3 or len({case['sample_id'] for case in cases}) != 3:
        raise ValueError('No-TC requires exactly three unique fixed joint cases')
    for case, original in zip(cases, originals):
        case['image_path'] = original['image_path']
    return cases


def no_tc_validate_group(plan, jobs, kind, code_hashes):
    teacher = kind == NO_TC_TEACHER_KIND
    if (plan.get('kind') != kind or plan.get('no_text_encoder') is not True
            or plan.get('text_encoder_loaded') is not False
            or plan.get('semantic_student_prompt', 'missing') is not None
            or not no_tc_count(plan.get('prefix_length'), 0)
            or plan.get('conditions') != ['joint'] or not no_tc_count(plan.get('videos'), 3)
            or plan.get('no_training') is not True or plan.get('student_uses_caption') is not False
            or plan.get('zero_music_input') is not False or plan.get('legacy_clip_complete_chain') is not False
            or plan.get('old_model_not_trained_for_no_tc') is not (not teacher)
            or jobs.get('no_text_encoder') is not True
            or jobs.get('old_model_not_trained_for_no_tc') is not (not teacher)
            or jobs.get('all_sampler_jobs_exit_zero') is not True
            or jobs.get('native_silence_identical_across_all_music_and_images') is not True
            or not no_tc_count(jobs.get('expected'), 3)):
        raise ValueError('No-TC plan/jobs scope or counts do not match the declared three-joint diagnostic')
    if (any(not no_tc_hash(value) for value in code_hashes.values())
            or plan.get('sampler_sha256') != code_hashes['sampler_sha256']):
        raise ValueError('No-TC generation sampler identity differs from the frozen source release')
    records, reports = jobs.get('jobs'), jobs.get('condition_reports')
    if (not isinstance(records, list) or len(records) != 3
            or not isinstance(reports, list) or len(reports) != 3
            or any(not isinstance(row, dict) for row in records + reports)
            or {row.get('shard') for row in records} != {0, 1, 2}
            or {row.get('shard') for row in reports} != {0, 1, 2}
            or any(type(row.get('shard')) is not int or not no_tc_count(row.get('exit_code'), 0) for row in records)
            or any(type(row.get('shard')) is not int or row.get('status') != 'passed' for row in reports)):
        raise ValueError('No-TC jobs or condition reports are missing, duplicated or failed')


def no_tc_validate_records(inputs, condition, native, case, kind, source, deployment=None):
    teacher = kind == NO_TC_TEACHER_KIND
    length, shape = condition.get('predicted_length'), condition.get('tail_shape')
    if (kind not in (NO_TC_TEACHER_KIND, NO_TC_STUDENT_KIND)
            or inputs.get('semantic_text', 'missing') is not None
            or inputs.get('no_text_encoder') is not True or inputs.get('text_encoder_loaded') is not False
            or inputs.get('image_path') != case['image_path'] or inputs.get('audio_path') != case['audio_path']
            or not no_tc_count(inputs.get('seed'), 42) or inputs.get('native_audio_condition') != 'fixed_silence'
            or inputs.get('music_encoder_ckpt') is not None or inputs.get('text_hidden_oracle') is not False
            or inputs.get('text_hidden_projection') is not None or inputs.get('music_text_zero_input') is not False
            or inputs.get('legacy_clip_complete_chain') is not False
            or condition.get('kind') != kind or condition.get('loaded') is not False
            or condition.get('semantic_text', 'missing') is not None
            or condition.get('processor_loaded') is not False or condition.get('tokenizer_loaded') is not False
            or not no_tc_count(condition.get('calls'), 0) or not no_tc_count(condition.get('prefix_length'), 0)
            or type(length) is not int or length < 1 or not isinstance(shape, list)
            or len(shape) != 2 or any(type(value) is not int for value in shape)
            or shape != [length, 5120] or not no_tc_hash(condition.get('raw_sha256'))):
        raise ValueError('No-TC condition contains text processing, a prefix or a wrong fixed-case/raw-token route')
    if (native.get('native_music_bypass') is not False or native.get('all_target_audio_clean') is not True
            or native.get('text_encoder_loaded') is not False or native.get('tokenizer_loaded') is not False
            or native.get('processor_loaded') is not False or native.get('semantic_text', 'missing') is not None
            or not no_tc_count(native.get('text_encoder_calls'), 0)
            or not no_tc_count(native.get('prefix_tokens'), 0) or native.get('prompt_embeds_shape') != shape
            or any(type(value) is not int for value in native.get('prompt_embeds_shape', []))
            or type(native.get('visual_reference_rows')) is not int or native['visual_reference_rows'] < 1
            or not no_tc_hash(native.get('visual_reference_sha256'))
            or any(not no_tc_hash(native.get(key)) for key in NO_TC_NATIVE_KEYS)):
        raise ValueError('No-TC native isolation or independent visual VAE-reference evidence is invalid')
    if teacher:
        expected_case = dict(sample_id=case['sample_id'], audio_id=Path(case['audio_path']).stem,
                             condition='joint', image_path=case['image_path'], audio_path=case['audio_path'])
        if (inputs.get('music_text_ckpt') is not None or inputs.get('no_tc_teacher_cache') != source.get('cache_dir')
                or condition.get('teacher_only_diagnostic') is not True or condition.get('music_prediction') is not False
                or condition.get('student_reads_caption') is not False
                or condition.get('context_source') != 'none_teacher_tail_only_diagnostic'
                or not no_tc_count(condition.get('original_teacher_tokens'), length)
                or source.get('case') != expected_case or not no_tc_count(source.get('tail_length'), length)
                or source.get('source_path') != str(Path(source.get('cache_dir', '')) / (Path(case['filename']).stem + '.pt'))
                or any(condition.get(key) != expected_case[key] for key in ('sample_id', 'image_path', 'audio_path'))
                or any(not no_tc_hash(source.get(key)) for key in ('source_sha256', 'teacher_tail_sha256', 'raw_sha256'))
                or any(condition.get(key) != source[key] for key in ('source_path', 'source_sha256', 'teacher_tail_sha256', 'raw_sha256'))
                or deployment is not None):
            raise ValueError('No-TC cached teacher source, tail hash, original length or joint-case identity changed')
    else:
        if (inputs.get('music_text_ckpt') != source.get('checkpoint') or inputs.get('no_tc_teacher_cache') is not None
                or inputs.get('waveform_loader') != 'music_clip_bridge.load_audio'
                or condition.get('old_model_not_trained_for_no_tc') is not True
                or condition.get('uses_ground_truth_caption') is not False
                or condition.get('context_source') != 'zeros_legacy_checkpoint_compatibility'
                or source.get('checkpoint_sha256') != NO_TC_C300_SHA
                or condition.get('checkpoint_sha256') != source['checkpoint_sha256']
                or source.get('complete_full_raw_checkpoint') is not True
                or type(source.get('query_max')) is not int or not 1 <= length <= source['query_max']
                or not isinstance(deployment, dict) or deployment.get('checkpoint_sha256') != source['checkpoint_sha256']
                or deployment.get('tail_shape') != shape or not no_tc_count(deployment.get('predicted_length'), length)
                or not no_tc_count(deployment.get('call'), 1)
                or deployment.get('final_adapter_projection_executed') is not True
                or deployment.get('uses_ground_truth_caption') is not False
                or deployment.get('legacy_clip_complete_chain') is not False
                or not no_tc_hash(deployment.get('input_sha256'))):
            raise ValueError('No-TC old student does not use its own length and unchanged complete raw checkpoint')


def no_tc_validate_report(report, cases, directories):
    tags = (NO_TC_TEACHER_TAG, NO_TC_STUDENT_TAG)
    if (report.get('complete') is not True or report.get('no_tc_complete') is not True
            or any(not no_tc_count(report.get(key), 6) for key in ('expected', 'passed', 'expected_new', 'passed_new'))
            or not no_tc_count(report.get('baseline_reused_new_count'), 0)
            or report.get('all_native_silence_hashes_identical') is not True
            or report.get('same_case_visual_reference_hashes_identical') is not True
            or report.get('complete_selected_audio_video_decode') is not True
            or report.get('musical_gain_accepted') is not False or report.get('new_interface_trained') is not False
            or not no_tc_count(report.get('optimizer_updates'), 0)
            or not isinstance(report.get('rows'), list) or len(report['rows']) != 6
            or any(not isinstance(row, dict) for row in report['rows'])
            or len(cases) != 3 or len({case['sample_id'] for case in cases}) != 3
            or set(directories) != set(tags)
            or report.get('generation_bindings') != [dict(tag=tag, directory=str(directories[tag]),
                kind=NO_TC_TEACHER_KIND if tag == NO_TC_TEACHER_TAG else NO_TC_STUDENT_KIND) for tag in tags]):
        raise ValueError('No-TC final report is incomplete or has the wrong diagnostic scope/counts/tags')
    rows = report['rows']
    expected = {(tag, case['sample_id'], 'joint', str(directories[tag] / 'joint' / case['filename']), case['audio_path'])
                for tag in tags for case in cases}
    if {(row.get('tag'), row.get('sample_id'), row.get('condition'), row.get('video_path'), row.get('audio_path'))
            for row in rows} != expected:
        raise ValueError('No-TC report does not bind exactly the six fixed joint outputs')
    for row in rows:
        case = next(case for case in cases if case['sample_id'] == row['sample_id'])
        kind = NO_TC_TEACHER_KIND if row['tag'] == NO_TC_TEACHER_TAG else NO_TC_STUDENT_KIND
        no_tc_validate_records(row['inputs'], row['no_tc_condition'], row['native_audio'], case, kind,
                               row['source'], row.get('deployment'))
        integrity = row.get('integrity', {})
        streams = integrity.get('streams', [])
        if (integrity.get('full_video_audio_decode') is not True
                or not no_tc_count(integrity.get('ffmpeg_decoded_video_frames'), 124)
                or not any(stream.get('codec_type') == 'video' and no_tc_count(stream.get('width'), 704)
                    and no_tc_count(stream.get('height'), 1280) and no_tc_count(stream.get('decoded_frames'), 124)
                    and type(stream.get('fps')) in (int, float) and stream['fps'] == 24 for stream in streams)
                or not any(stream.get('codec_type') == 'audio' for stream in streams)
                or any(not no_tc_hash(row.get(key)) for key in ('video_sha256', 'log_sha256', 'image_sha256', 'audio_sha256'))
                or row.get('diagnostic_only') is not True or row.get('musical_quality_accepted') is not False):
            raise ValueError('No-TC row lacks complete 124-frame704p24fps media/hash evidence')
    if any(len({row['native_audio'][key] for row in rows}) != 1 for key in NO_TC_NATIVE_KEYS):
        raise ValueError('All six No-TC native silence hashes must be identical')
    for case in cases:
        pair = [row for row in rows if row['sample_id'] == case['sample_id']]
        if any(len({row[key] for row in pair}) != 1 for key in ('image_sha256', 'audio_sha256')) or any(
                len({row['native_audio'][key] for row in pair}) != 1
                for key in ('visual_reference_rows', 'visual_reference_sha256')):
            raise ValueError('No-TC same-case independent VAE visual reference or source image/audio changed')
    for key in ('sampler_sha256', 'infer_sha256', 'diagnostic_sha256'):
        if not no_tc_hash(report.get('code_sha256', {}).get(key)):
            raise ValueError('No-TC report lacks code identity: ' + key)


def no_tc_teacher_source(condition, case):
    import hashlib
    import torch
    path = Path(condition['source_path'])
    before = signature(path)
    record = torch.load(path, map_location='cpu', weights_only=True)
    tail, metadata = record['tail'], record['metadata']
    if (tail.ndim != 2 or tail.shape[1] != 5120 or not torch.isfinite(tail).all()
            or metadata.get('kind') != 'oracle_teacher_hidden_not_music_prediction'
            or not no_tc_count(metadata.get('tail_length'), len(tail))):
        raise ValueError('No-TC cached original teacher tail is invalid')
    content = tail.contiguous().view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256(str(tail.dtype).encode() + str(tuple(tail.shape)).encode() + content).hexdigest()
    deployed_sha = hashlib.sha256(tail.bfloat16().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    if metadata.get('teacher_tail_sha256') != digest or signature(path) != before:
        raise ValueError('No-TC cached teacher tail identity changed')
    source_case = metadata['case']
    source = dict(cache_dir=str(path.parent), source_path=str(path), source_sha256=sha256(path),
        teacher_tail_sha256=digest, raw_sha256=deployed_sha, tail_length=len(tail),
        case={key: source_case.get(key) for key in ('sample_id', 'audio_id', 'condition', 'image_path', 'audio_path')})
    expected_path = path.parent / (Path(case['filename']).stem + '.pt')
    if path != expected_path:
        raise ValueError('No-TC cached teacher path is not the fixed joint case')
    return source


def no_tc_audit_main():
    import decord
    import torch
    from PIL import Image, ImageDraw
    parser = argparse.ArgumentParser(description='Strict six-fixed-joint no-TC diagnostic audit; decoding is not quality acceptance')
    parser.add_argument('--no_tc_audit', action='store_true', required=True)
    parser.add_argument('--exp_root', type=Path, required=True)
    parser.add_argument('--teacher_dir', type=Path, required=True)
    parser.add_argument('--student_dir', type=Path, required=True)
    parser.add_argument('--release_report', type=Path, required=True, help='Frozen generation-source release; never retrofit fields into running logs')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not args.out.parent.is_dir():
        parser.error('Require a new report and existing parent')
    exp, cases = args.exp_root, no_tc_cases(args.exp_root)
    directories = {NO_TC_TEACHER_TAG: args.teacher_dir, NO_TC_STUDENT_TAG: args.student_dir}
    if any(path.resolve() != (exp / 'H3AR' / ('val_musicreplace_' + tag)).resolve()
           for tag, path in directories.items()):
        parser.error('No-TC directory names must bind text_no_tc_teacher_v1 and text_no_tc_C300_v1')
    release_sha = sha256(args.release_report)
    release = json.loads(args.release_report.read_text())
    if (release.get('release_complete') is not True or not no_tc_count(release.get('expected_new_videos'), 6)
            or release.get('source_checkpoint', {}).get('sha256') != NO_TC_C300_SHA
            or not isinstance(release.get('groups'), list) or len(release['groups']) != 2):
        raise ValueError('Require the exact six-video source release and original C300 identity')
    sources = release.get('source_files', {})
    if set(sources) != {'sampler', 'infer', 'manager'} or any(
            not no_tc_hash(value.get('sha256')) or sha256(Path(value['path'])) != value['sha256']
            for value in sources.values()):
        raise ValueError('Frozen generation source identity changed')
    for tag, directory in directories.items():
        kind = NO_TC_TEACHER_KIND if tag == NO_TC_TEACHER_TAG else NO_TC_STUDENT_KIND
        bound = [group for group in release['groups'] if group.get('kind') == kind]
        if len(bound) != 1 or bound[0].get('directory') != str(directory) or not no_tc_count(bound[0].get('expected_videos'), 3):
            raise ValueError('Source release does not bind the two exact generation directories')
    sampler, infer = Path(sources['sampler']['path']), Path(sources['infer']['path'])
    code_hashes = dict(sampler_sha256=sources['sampler']['sha256'], infer_sha256=sources['infer']['sha256'])
    rows, artifacts, pending_images, pending_markers = [], [dict(path=str(args.release_report), sha256=release_sha)] + [
        dict(path=value['path'], sha256=value['sha256']) for value in sources.values()], [], []
    for tag, directory in directories.items():
        teacher = tag == NO_TC_TEACHER_TAG
        kind = NO_TC_TEACHER_KIND if teacher else NO_TC_STUDENT_KIND
        marker, mosaic = directory / 'no_tc_integrity.json', directory / 'no_tc_joint_mosaic.png'
        if marker.exists() or mosaic.exists():
            raise FileExistsError('Refuse existing No-TC marker or mosaic')
        plan_path, jobs_path = directory / 'plan.json', directory / 'jobs.json'
        plan, jobs = json.loads(plan_path.read_text()), json.loads(jobs_path.read_text())
        no_tc_validate_group(plan, jobs, kind, code_hashes)
        artifacts += [dict(path=str(path), sha256=sha256(path)) for path in (plan_path, jobs_path)]
        checkpoint_source = None
        if not teacher:
            first_job = next(row for row in jobs['jobs'] if row['shard'] == 0)
            first_inputs = one_record(Path(first_job['log']).read_text(), '[h3ar] input_conditions ')
            checkpoint = Path(first_inputs['music_text_ckpt'])
            summary_path = checkpoint.parent / 'finished.json'
            summary = json.loads(summary_path.read_text())
            checkpoint_sha = sha256(checkpoint)
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            metadata, weights = state['metadata'], state['model']
            if (summary.get('complete') is not True or not no_tc_count(summary.get('steps'), 300)
                    or checkpoint_sha != NO_TC_C300_SHA or summary.get('checkpoint_sha256') != checkpoint_sha
                    or metadata.get('state_scope') != 'music_to_h3_text' or metadata.get('format_version') != 2
                    or metadata.get('backend') != 'musicfm' or metadata.get('raw_dim') != 5120
                    or not any(key.startswith('backbone.') for key in weights)
                    or not any(key.startswith('adapter.length_head.') for key in weights)
                    or not any(key.startswith('adapter.raw_output.') for key in weights)
                    or plan.get('checkpoint_sha256') != checkpoint_sha
                    or release['source_checkpoint'].get('path') != str(checkpoint)):
                raise ValueError('No-TC student source is not the unchanged complete old C300 backbone/raw-adapter model')
            checkpoint_source = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                checkpoint_summary=str(summary_path), checkpoint_summary_sha256=sha256(summary_path),
                query_max=metadata['query_max'], complete_full_raw_checkpoint=True,
                old_model_not_trained_for_no_tc=True)
            artifacts += [dict(path=str(checkpoint), sha256=checkpoint_sha),
                          dict(path=str(summary_path), sha256=sha256(summary_path))]
            del state, weights
        grid = Image.new('RGB', (176 * 7, 348 * 3), 'white')
        draw = ImageDraw.Draw(grid)
        group_rows = []
        for index, case in enumerate(cases):
            evidence = successful_log(directory / 'logs', case, require_replacement=False, require_empty_prompt=False)
            if evidence is None:
                raise ValueError('Missing exact successful no-TC joint log: ' + case['filename'])
            log_path, log_sha, log_signature, _ = evidence
            text = log_path.read_text()
            header = json.loads(text.splitlines()[0])
            job = next(row for row in jobs['jobs'] if row['shard'] == case['shard'])
            job_report = next(row for row in jobs['condition_reports'] if row['shard'] == case['shard'])
            command = header.get('command', [])
            if (job.get('log') != str(log_path) or header.get('sampler_sha256') != code_hashes['sampler_sha256']
                    or '--no_text_encoder' not in command or '--fixed_native_silence' not in command
                    or '--audio_override' in command or command.count('--conds') != 1
                    or command[command.index('--conds') + 1] != 'joint'):
                raise ValueError('No-TC successful log/job/source command identity mismatch')
            inputs = one_record(text, '[h3ar] input_conditions ')
            condition = one_record(text, '[no tc condition] ')
            native = one_record(text, '[no tc native isolation] ')
            deployment = None if teacher else one_record(text, '[music text deployment] ')
            source = no_tc_teacher_source(condition, case) if teacher else checkpoint_source.copy()
            no_tc_validate_records(inputs, condition, native, case, kind, source, deployment)
            route_flag = '--no_tc_teacher_cache' if teacher else '--music_text_ckpt'
            route_path = source['cache_dir'] if teacher else source['checkpoint']
            if (command.count(route_flag) != 1 or command[command.index(route_flag) + 1] != route_path
                    or ('--music_text_ckpt' in command if teacher else '--no_tc_teacher_cache' in command)):
                raise ValueError('No-TC logged launch command does not bind the exact source route')
            if (job_report.get('no_tc_condition') != condition or job_report.get('native_audio') != native
                    or (not teacher and job_report.get('deployment') != deployment)
                    or (teacher and (plan.get('teacher_tail_cache') != source['cache_dir']
                                     or next(group for group in release['groups'] if group['kind'] == kind).get('teacher_tail_cache') != source['cache_dir']))):
                raise ValueError('No-TC jobs evidence or declared teacher cache differs from successful log')
            video = directory / 'joint' / case['filename']
            before = signature(video)
            integrity = probe_video(video)
            reader = decord.VideoReader(str(video), num_threads=1)
            with Image.open(case['image_path']) as image:
                reference = image.convert('RGB')
            images = [reference] + [Image.fromarray(reader[frame].asnumpy()) for frame in (0, 24, 48, 72, 96, 123)]
            for column, image in enumerate(images):
                grid.paste(image.resize((176, 320)), (176 * column, 348 * index + 28))
                draw.text((176 * column + 3, 348 * index + 6), case['sample_id'] if column == 0
                          else str((0, 24, 48, 72, 96, 123)[column - 1] / 24) + 's', fill='black')
            row = dict(tag=tag, sample_id=case['sample_id'], condition='joint', audio_path=case['audio_path'],
                video_path=str(video), video_sha256=sha256(video), log_path=str(log_path), log_sha256=log_sha,
                image_sha256=sha256(Path(case['image_path'])), audio_sha256=sha256(Path(case['audio_path'])),
                inputs=inputs, no_tc_condition=condition, native_audio=native, deployment=deployment,
                source=source, integrity=integrity, diagnostic_only=True, musical_quality_accepted=False)
            if signature(video) != before or signature(log_path) != log_signature:
                raise ValueError('No-TC video/log changed during full-media audit')
            if teacher:
                artifacts.append(dict(path=source['source_path'], sha256=source['source_sha256']))
            artifacts += [dict(path=case['image_path'], sha256=row['image_sha256']),
                          dict(path=case['audio_path'], sha256=row['audio_sha256'])]
            group_rows.append(row)
            rows.append(row)
        pending_images.append((mosaic, grid))
        pending_markers.append((marker, dict(complete=True, no_tc_complete=True, expected=3, passed=3,
            tag=tag, kind=kind, rows=group_rows, mosaic=str(mosaic), final_no_tc_report=str(args.out),
            diagnostic_only=True, new_interface_trained=False, musical_quality_accepted=False)))
        print(json.dumps(dict(event='no_tc_group_full_decoded', tag=tag, passed=3)), flush=True)
    result = dict(complete=True, no_tc_complete=True, expected=6, passed=6, expected_new=6, passed_new=6,
        baseline_reused_new_count=0, reused_baselines=[str(exp / 'H3AR/val_musicreplace_text_oracle_neutral_v1'),
            str(exp / 'H3AR/val_musicreplace_text_online_C_len001_s300_v1')],
        generation_bindings=[dict(tag=tag, directory=str(directory),
            kind=NO_TC_TEACHER_KIND if tag == NO_TC_TEACHER_TAG else NO_TC_STUDENT_KIND)
            for tag, directory in directories.items()], rows=rows,
        mosaics=[str(path) for path, _ in pending_images], source_artifacts=artifacts,
        code_sha256=dict(code_hashes, diagnostic_sha256=sha256(Path(__file__))),
        generation_source_files=sources, release_report=str(args.release_report), release_report_sha256=release_sha,
        all_native_silence_hashes_identical=True, same_case_visual_reference_hashes_identical=True,
        complete_selected_audio_video_decode=True, optimizer_updates=0, new_interface_trained=False,
        musical_gain_accepted=False, audio_mux_match_checked=False, continuous_human_review='pending',
        scope='TC absent, no blank text or prefix; cached teacher tail only and OLD C300 zero-context diagnostic. '
            'Independent visual VAE references retained and hashes identical per case. GT music only muxed, '
            'not a native condition; source-audio/mux equality not independently checked. Full decode is integrity, '
            'not dance quality, music gain or successful training/deployment of a model trained for the new interface.')
    no_tc_validate_report(result, cases, directories)
    for artifact in artifacts:
        if sha256(Path(artifact['path'])) != artifact['sha256']:
            raise ValueError('No-TC source artifact changed during audit')
    if code_hashes != dict(sampler_sha256=sha256(sampler), infer_sha256=sha256(infer)):
        raise ValueError('No-TC generation code changed during audit')
    for path, image in pending_images:
        with path.open('xb') as stream:
            image.save(stream, format='PNG')
    atomic_write_new(args.out, result)
    report_sha = sha256(args.out)
    for path, record in pending_markers:
        record['final_no_tc_report_sha256'] = report_sha
        atomic_write_new(path, record)
    print(json.dumps(dict(event='no_tc_audit_finished', expected=6, passed=6, out=str(args.out))), flush=True)


def no_tc_fixture_main():
    import contextlib
    import copy
    import hashlib
    import io
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch
    import music_text_review as review
    parser = argparse.ArgumentParser(description='Pure-memory CPU No-TC acceptance/rejection and unchanged-default review fixtures')
    parser.add_argument('--no_tc_fixture', action='store_true', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not args.out.parent.is_dir():
        parser.error('Require a new CPU fixture JSON and existing parent')
    digest = lambda name: hashlib.sha256(name.encode()).hexdigest()
    exp = Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp')
    report_path = Path('/fixture/no_tc_report.json')
    tags = (NO_TC_TEACHER_TAG, NO_TC_STUDENT_TAG)
    directories = {tag: exp / 'H3AR' / ('val_musicreplace_' + tag) for tag in tags}
    cases = [dict(sample_id='sid' + str(index), condition='joint', shard=index,
        image_path='/fixture/image' + str(index) + '.png', audio_path='/fixture/sid' + str(index) + '.wav',
        filename='sid' + str(index) + '_sid' + str(index) + '.mp4') for index in range(3)]
    code_hashes = dict(sampler_sha256=digest('sampler'), infer_sha256=digest('infer'))
    rows, groups = [], []
    for tag in tags:
        teacher = tag == NO_TC_TEACHER_TAG
        kind = NO_TC_TEACHER_KIND if teacher else NO_TC_STUDENT_KIND
        plan = dict(kind=kind, no_text_encoder=True, text_encoder_loaded=False, semantic_student_prompt=None,
            prefix_length=0, conditions=['joint'], videos=3, no_training=True, student_uses_caption=False,
            zero_music_input=False, legacy_clip_complete_chain=False, old_model_not_trained_for_no_tc=not teacher,
            sampler_sha256=code_hashes['sampler_sha256'])
        jobs = dict(no_text_encoder=True,
            old_model_not_trained_for_no_tc=not teacher, all_sampler_jobs_exit_zero=True,
            native_silence_identical_across_all_music_and_images=True, expected=3,
            jobs=[dict(shard=i, exit_code=0) for i in range(3)],
            condition_reports=[dict(shard=i, status='passed') for i in range(3)])
        no_tc_validate_group(plan, jobs, kind, code_hashes)
        groups.append((plan, jobs, kind))
        for case in cases:
            length = 3 if teacher else 2
            source = dict(cache_dir='/fixture/cache', source_path='/fixture/cache/' + Path(case['filename']).stem + '.pt',
                source_sha256=digest(case['sample_id'] + 'source'), teacher_tail_sha256=digest(case['sample_id'] + 'tail'),
                raw_sha256=digest(case['sample_id'] + 'raw'), tail_length=length,
                case=dict(sample_id=case['sample_id'], audio_id=case['sample_id'], condition='joint',
                          image_path=case['image_path'], audio_path=case['audio_path'])) if teacher else dict(
                checkpoint=str(exp / 'MS/music_text_online_C_len001_s300_v1/last.pt'),
                checkpoint_sha256=NO_TC_C300_SHA, query_max=8, complete_full_raw_checkpoint=True)
            inputs = dict(semantic_text=None, no_text_encoder=True, text_encoder_loaded=False,
                image_path=case['image_path'], audio_path=case['audio_path'], seed=42,
                native_audio_condition='fixed_silence', music_encoder_ckpt=None, text_hidden_oracle=False,
                text_hidden_projection=None, music_text_zero_input=False, legacy_clip_complete_chain=False,
                music_text_ckpt=None if teacher else source['checkpoint'],
                no_tc_teacher_cache=source['cache_dir'] if teacher else None,
                waveform_loader='librosa' if teacher else 'music_clip_bridge.load_audio')
            condition = dict(kind=kind, loaded=False, semantic_text=None, processor_loaded=False, tokenizer_loaded=False,
                calls=0, prefix_length=0, predicted_length=length, tail_shape=[length, 5120],
                raw_sha256=source['raw_sha256'] if teacher else digest(case['sample_id'] + 'student_raw'))
            if teacher:
                condition.update(teacher_only_diagnostic=True, music_prediction=False, student_reads_caption=False,
                    context_source='none_teacher_tail_only_diagnostic', original_teacher_tokens=length,
                    sample_id=case['sample_id'], image_path=case['image_path'], audio_path=case['audio_path'],
                    **{key: source[key] for key in ('source_path', 'source_sha256', 'teacher_tail_sha256')})
            else:
                condition.update(old_model_not_trained_for_no_tc=True, uses_ground_truth_caption=False,
                    context_source='zeros_legacy_checkpoint_compatibility', checkpoint_sha256=source['checkpoint_sha256'])
            native = dict(native_music_bypass=False, all_target_audio_clean=True, text_encoder_loaded=False,
                tokenizer_loaded=False, processor_loaded=False, semantic_text=None, text_encoder_calls=0,
                prefix_tokens=0, prompt_embeds_shape=[length, 5120], visual_reference_rows=1,
                visual_reference_sha256=digest(case['sample_id'] + 'visual'),
                **{key: digest(key) for key in NO_TC_NATIVE_KEYS})
            deployment = None if teacher else dict(checkpoint_sha256=source['checkpoint_sha256'],
                tail_shape=[length, 5120], predicted_length=length, call=1, final_adapter_projection_executed=True,
                uses_ground_truth_caption=False, legacy_clip_complete_chain=False, input_sha256=digest('proxy_input'))
            rows.append(dict(tag=tag, sample_id=case['sample_id'], condition='joint', audio_path=case['audio_path'],
                video_path=str(directories[tag] / 'joint' / case['filename']), video_sha256=digest(tag + case['sample_id']),
                log_path='/fixture/log', log_sha256=digest('log'), image_sha256=digest(case['sample_id'] + 'image'),
                audio_sha256=digest(case['sample_id'] + 'audio'), inputs=inputs, no_tc_condition=condition,
                native_audio=native, deployment=deployment, source=source, diagnostic_only=True,
                musical_quality_accepted=False, integrity=dict(full_video_audio_decode=True,
                    ffmpeg_decoded_video_frames=124, streams=[dict(codec_type='video', width=704, height=1280,
                        decoded_frames=124, fps=24), dict(codec_type='audio')])))
    report = dict(complete=True, no_tc_complete=True, expected=6, passed=6, expected_new=6, passed_new=6,
        baseline_reused_new_count=0, all_native_silence_hashes_identical=True,
        same_case_visual_reference_hashes_identical=True, complete_selected_audio_video_decode=True,
        musical_gain_accepted=False, new_interface_trained=False, optimizer_updates=0, rows=rows,
        generation_bindings=[dict(tag=tag, directory=str(directories[tag]),
            kind=NO_TC_TEACHER_KIND if tag == NO_TC_TEACHER_TAG else NO_TC_STUDENT_KIND) for tag in tags],
        code_sha256=dict(code_hashes, diagnostic_sha256=digest('diagnostic')))
    no_tc_validate_report(report, cases, directories)
    markers = {tag: dict(complete=True, no_tc_complete=True, tag=tag,
        kind=NO_TC_TEACHER_KIND if tag == NO_TC_TEACHER_TAG else NO_TC_STUDENT_KIND, expected=3, passed=3,
        rows=[row for row in rows if row['tag'] == tag], final_no_tc_report=str(report_path),
        final_no_tc_report_sha256=digest('report'), diagnostic_only=True, new_interface_trained=False,
        musical_quality_accepted=False) for tag in tags}
    base_ids = ['H3AR/val_musicreplace_text_oracle_neutral_v1', 'H3AR/val_musicreplace_text_online_C_len001_s300_v1']
    base = dict(exp=str(exp), models=[dict(id=name, desc='unchanged', scores={'quality': 2}) for name in
        base_ids + ['old_model_' + str(i) for i in range(35)]], samples=[case['sample_id'] for case in cases],
        presets=[dict(name='old_presets', models=base_ids)], scores={'neutral_person': 1}, scope='old scope')
    original = copy.deepcopy(base)
    manifest = review.no_tc_review_manifest(base, report, cases, directories, markers, report_path, digest('report'))
    assert base == original and manifest['models'][:37] == base['models'] and len(manifest['models']) == 39
    assert manifest['presets'][2:] == base['presets'] and manifest['samples'] == base['samples']
    assert manifest['scores'] == base['scores']
    assert all(set(model['conds']) == {'joint'} and len(model['conds']['joint']) == 3 for model in manifest['models'][37:])
    rejected = []

    def reject_report(name, path, value):
        candidate = copy.deepcopy(report)
        target = candidate
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        try:
            no_tc_validate_report(candidate, cases, directories)
        except (ValueError, KeyError, TypeError):
            rejected.append(name)
        else:
            raise AssertionError('Accepted forbidden fixture: ' + name)

    for name, path, value in [
        ('semantic_blank', ('rows', 0, 'inputs', 'semantic_text'), ' '),
        ('semantic_empty', ('rows', 3, 'inputs', 'semantic_text'), ''),
        ('condition_semantic_blank', ('rows', 0, 'no_tc_condition', 'semantic_text'), ' '),
        ('TC_calls_bool', ('rows', 0, 'no_tc_condition', 'calls'), False),
        ('prefix_bool', ('rows', 0, 'no_tc_condition', 'prefix_length'), False),
        ('TC_loaded', ('rows', 3, 'inputs', 'text_encoder_loaded'), True),
        ('native_TC_calls_bool', ('rows', 0, 'native_audio', 'text_encoder_calls'), False),
        ('native_prefix_bool', ('rows', 0, 'native_audio', 'prefix_tokens'), False),
        ('no_visual', ('rows', 0, 'native_audio', 'visual_reference_rows'), 0),
        ('visual_rows_bool', ('rows', 0, 'native_audio', 'visual_reference_rows'), True),
        ('visual_hash_wrong', ('rows', 0, 'native_audio', 'visual_reference_sha256'), digest('changed_visual')),
        ('native_hash_wrong', ('rows', 0, 'native_audio', 'mask_sha256'), digest('changed_mask')),
        ('teacher_source_hash_wrong', ('rows', 0, 'source', 'source_sha256'), digest('changed_source')),
        ('teacher_tail_hash_wrong', ('rows', 0, 'source', 'teacher_tail_sha256'), digest('changed_tail')),
        ('teacher_source_sid_wrong', ('rows', 0, 'source', 'case', 'sample_id'), 'wrong_sid'),
        ('row_sid_wrong', ('rows', 0, 'sample_id'), 'wrong_sid'),
        ('count_wrong', ('passed',), 5), ('count_bool', ('passed',), True),
        ('tag_wrong', ('rows', 0, 'tag'), 'text_oracle_neutral_v1'),
        ('student_length_wrong', ('rows', 3, 'deployment', 'predicted_length'), 3),
        ('student_checkpoint_wrong', ('rows', 3, 'deployment', 'checkpoint_sha256'), digest('wrong_checkpoint')),
        ('student_zero_context_wrong', ('rows', 3, 'no_tc_condition', 'context_source'), 'TC_prefix'),
        ('incomplete_media', ('rows', 0, 'integrity', 'ffmpeg_decoded_video_frames'), 123),
        ('same_image_hash_wrong', ('rows', 0, 'image_sha256'), digest('changed_image')),
    ]:
        reject_report(name, path, value)
    for field, value in [('passed', 2), ('passed', True), ('final_no_tc_report_sha256', digest('wrong_report')),
                         ('rows', []), ('tag', NO_TC_STUDENT_TAG)]:
        wrong = copy.deepcopy(markers)
        wrong[NO_TC_TEACHER_TAG][field] = value
        try:
            review.no_tc_review_manifest(base, report, cases, directories, wrong, report_path, digest('report'))
        except ValueError:
            rejected.append('review_marker_' + field + '_' + str(type(value).__name__))
        else:
            raise AssertionError('Accepted wrong No-TC review marker')
    plan, jobs, kind = groups[0]
    for field, value in [('expected', 4), ('expected', True), ('no_text_encoder', False), ('jobs', [])]:
        wrong = copy.deepcopy(jobs)
        wrong[field] = value
        try:
            no_tc_validate_group(plan, wrong, kind, code_hashes)
        except ValueError:
            rejected.append('jobs_' + field + '_' + str(type(value).__name__))
        else:
            raise AssertionError('Accepted wrong No-TC jobs')
    log_text = '[h3ar] input_conditions ' + json.dumps(rows[0]['inputs']) + '\n'
    log_text += '[h3ar] repeat_clean audio tail: 4 rows fixed at effective time=1\n[h3ar] joint sid0 1s\n[h3ar] done\n'
    log_dir, log = MagicMock(), MagicMock()
    log_dir.__truediv__.return_value = log
    log.exists.return_value = True
    log.stat.return_value = SimpleNamespace(st_dev=1, st_ino=2, st_size=len(log_text), st_mtime_ns=3)
    log.read_bytes.return_value = log_text.encode()
    assert successful_log(log_dir, cases[0], require_replacement=False, require_empty_prompt=False) is not None
    captured = []
    default_base = copy.deepcopy(base)
    default_base['models'][0]['id'] = 'default_old_baseline'
    all_cases = cases + [dict(case, condition='aswap', shard=case['shard'] + 3) for case in cases]
    def memory_read(path, *unused_args, **unused_kwargs):
        return json.dumps(default_base if str(path) == '/fixture/base.json' else dict(complete=True))
    with patch.object(sys, 'argv', ['review', '--base_manifest', '/fixture/base.json', '--out', '/fixture/default.json']), \
            patch.object(Path, 'read_text', memory_read), patch.object(Path, 'exists', return_value=False), \
            patch.object(Path, 'is_dir', return_value=True), patch.object(Path, 'is_file', return_value=True), \
            patch.object(review, 'fixed_cases', return_value=all_cases), \
            patch.object(review, 'atomic_write_new', side_effect=lambda path, result: captured.append(result)), \
            contextlib.redirect_stdout(io.StringIO()):
        review.main()
    default = captured[0]
    assert len(default['models']) == 40 and default['models'][:37] == default_base['models']
    assert [row['id'] for row in default['models'][37:]] == ['H3AR/val_musicreplace_' + tag for tag in
        ('text_legacy_fullclip_v1', 'text_oracle_v1', 'text_oracle_neutral_v1')]
    assert default['presets'][2:] == base['presets'] and default['scores'] == base['scores']
    # Exercise the real audit dispatcher with in-memory sources/media, including publication ordering.
    from types import ModuleType
    files, file_hashes = {}, {}
    checkpoint = exp / 'MS/music_text_online_C_len001_s300_v1/last.pt'
    summary_path = checkpoint.parent / 'finished.json'
    files[summary_path] = json.dumps(dict(complete=True, steps=300, checkpoint_sha256=NO_TC_C300_SHA))
    file_hashes[checkpoint], file_hashes[summary_path] = NO_TC_C300_SHA, digest('summary')
    for tag, (plan, jobs, kind) in zip(tags, groups):
        directory = directories[tag]
        plan = copy.deepcopy(plan)
        jobs = copy.deepcopy(jobs)
        teacher = tag == NO_TC_TEACHER_TAG
        plan['checkpoint_sha256'] = None if teacher else NO_TC_C300_SHA
        plan['teacher_tail_cache'] = '/fixture/cache' if teacher else None
        for index, case in enumerate(cases):
            row = next(row for row in rows if row['tag'] == tag and row['sample_id'] == case['sample_id'])
            log_path = directory / 'logs' / ('shard_' + str(index) + '.log')
            command = ['--no_text_encoder', '--fixed_native_silence', '--conds', 'joint',
                       '--no_tc_teacher_cache' if teacher else '--music_text_ckpt',
                       '/fixture/cache' if teacher else str(checkpoint)]
            text = json.dumps(dict(command=command, sampler_sha256=code_hashes['sampler_sha256'])) + '\n'
            for prefix, value in [('[h3ar] input_conditions ', row['inputs']),
                                  ('[no tc condition] ', row['no_tc_condition']),
                                  ('[no tc native isolation] ', row['native_audio'])]:
                text += prefix + json.dumps(value) + '\n'
            if not teacher:
                text += '[music text deployment] ' + json.dumps(row['deployment']) + '\n'
            files[log_path] = text
            jobs['jobs'][index]['log'] = str(log_path)
            jobs['condition_reports'][index].update(no_tc_condition=row['no_tc_condition'], native_audio=row['native_audio'])
            if not teacher:
                jobs['condition_reports'][index]['deployment'] = row['deployment']
            if teacher:
                file_hashes[Path(row['source']['source_path'])] = row['source']['source_sha256']
            file_hashes[Path(case['image_path'])] = row['image_sha256']
            file_hashes[Path(case['audio_path'])] = row['audio_sha256']
        for name, value in [('plan.json', plan), ('jobs.json', jobs)]:
            files[directory / name] = json.dumps(value)
    fake_torch, fake_decord, fake_pil = ModuleType('torch'), ModuleType('decord'), ModuleType('PIL')
    fake_torch.load = lambda *a, **kw: dict(metadata=dict(state_scope='music_to_h3_text', format_version=2,
        backend='musicfm', raw_dim=5120, query_max=8), model={'backbone.weight': True,
        'adapter.length_head.weight': True, 'adapter.raw_output.weight': True})
    reader = MagicMock()
    reader.__getitem__.return_value.asnumpy.return_value = None
    fake_decord.VideoReader = lambda *a, **kw: reader
    image, draw = MagicMock(), MagicMock()
    fake_pil.Image = SimpleNamespace(new=lambda *a, **kw: image, open=lambda *a, **kw: image,
                                    fromarray=lambda *a, **kw: image)
    fake_pil.ImageDraw = SimpleNamespace(Draw=lambda *a, **kw: draw)
    image.__enter__.return_value = image
    image.convert.return_value = image
    image.resize.return_value = image
    module = sys.modules[__name__]
    def memory_sha(path):
        if Path(path).name == 'sample_h3_ar.py': return code_hashes['sampler_sha256']
        if Path(path).name == 'music_text_infer.py': return code_hashes['infer_sha256']
        if Path(path).name == 'music_text_diagnostic.py': return digest('diagnostic')
        return file_hashes.get(Path(path), digest(str(path)))
    def memory_log(directory, case, **kwargs):
        assert kwargs == dict(require_replacement=False, require_empty_prompt=False)
        path = directory / ('shard_' + str(case['shard']) + '.log')
        return path, digest(str(path)), (1, 2, 3, 4), {}
    def memory_teacher(condition, case):
        return copy.deepcopy(next(row['source'] for row in rows if row['tag'] == NO_TC_TEACHER_TAG
                                  and row['sample_id'] == case['sample_id']))
    audit_out = exp / 'MS/no_tc_audit_fixture.json'
    release_path = exp / 'MS/music_text_no_tc_v1_release.json'
    release = dict(release_complete=True, expected_new_videos=6,
        source_files={name: dict(path='/fixture/' + filename, sha256=digest(name)) for name, filename in
            [('sampler', 'sample_h3_ar.py'), ('infer', 'music_text_infer.py'), ('manager', 'music_text_oracle_generate.py')]},
        source_checkpoint=dict(path=str(checkpoint), sha256=NO_TC_C300_SHA, old_model_not_trained_for_no_tc=True),
        groups=[dict(kind=kind, directory=str(directories[tag]), expected_videos=3,
                     **({'teacher_tail_cache': '/fixture/cache'} if tag == NO_TC_TEACHER_TAG else {}))
                for tag, (_, _, kind) in zip(tags, groups)])
    files[release_path] = json.dumps(release)
    file_hashes[Path('/fixture/music_text_oracle_generate.py')] = digest('manager')
    audit_argv = ['diagnostic', '--no_tc_audit', '--exp_root', str(exp), '--teacher_dir', str(directories[tags[0]]),
                  '--student_dir', str(directories[tags[1]]), '--release_report', str(release_path), '--out', str(audit_out)]
    published, probes = [], []
    def run_flow(fail_last=False):
        def memory_probe(path):
            probes.append(str(path))
            if fail_last and len(probes) == 6:
                raise ValueError('fixture last full-media decode failure')
            return copy.deepcopy(rows[0]['integrity'])
        with patch.dict(sys.modules, {'torch': fake_torch, 'decord': fake_decord, 'PIL': fake_pil}), \
                patch.object(sys, 'argv', audit_argv), patch.object(Path, 'read_text', lambda path: files[path]), \
                patch.object(Path, 'exists', return_value=False), patch.object(Path, 'is_dir', return_value=True), \
                patch.object(Path, 'open', return_value=MagicMock()), patch.object(module, 'no_tc_cases', return_value=cases), \
                patch.object(module, 'sha256', memory_sha), patch.object(module, 'signature', return_value=(1, 2, 3, 4)), \
                patch.object(module, 'successful_log', memory_log), patch.object(module, 'no_tc_teacher_source', memory_teacher), \
                patch.object(module, 'probe_video', memory_probe), \
                patch.object(module, 'atomic_write_new', side_effect=lambda path, value: published.append((path, value))), \
                contextlib.redirect_stdout(io.StringIO()):
            main()
    run_flow()
    assert len(probes) == 6 and len(published) == 3 and published[0][0] == audit_out
    assert [path.name for path, _ in published[1:]] == ['no_tc_integrity.json', 'no_tc_integrity.json']
    no_tc_validate_report(published[0][1], cases, directories)
    published.clear(); probes.clear()
    try:
        run_flow(fail_last=True)
    except ValueError:
        assert len(probes) == 6 and not published
    else:
        raise AssertionError('A last-video decode failure published No-TC report/markers')
    rejected.append('last_decode_failure_no_markers')
    result = dict(complete=True, cpu_only=True, pure_memory_fixtures=True, gpu_started=False,
        real_video_audio_decode_executed=False, expected_fixture_rows=6, passed_fixture_rows=6,
        strict_rejections=rejected, strict_rejection_count=len(rejected),
        semantic_none_success_log_accepted_without_empty_prompt_helper=True,
        base37_models_preserved=True, no_tc_review_models=39, no_tc_review_only_joint3_per_model=True,
        old_presets_samples_scores_unchanged=True, legacy_default_mode_unchanged=True,
        actual_no_tc_audit_dispatch_full6_cpu_mock=True, final_report_published_before_two_markers=True,
        last_decode_failure_publishes_no_report_or_markers=True,
        fixture_report=report, fixture_review_new_models=manifest['models'][37:],
        source_sha256=dict(diagnostic=sha256(Path(__file__)), review=sha256(Path(review.__file__))),
        reproduce_command='python3 -B ' + str(Path(__file__).resolve()) + ' --no_tc_fixture --out '
            + str(args.out.with_name(args.out.stem + '_rerun.json')),
        scope='Pure-memory audit/review gate fixtures only; media success flags are synthetic and do not prove real decode, generation, VAE execution, quality or music gain')
    atomic_write_new(args.out, result)
    print(json.dumps(dict(event='no_tc_cpu_fixture_finished', passed=6, rejected=len(rejected), out=str(args.out))), flush=True)


def main():
    if '--no_tc_fixture' in __import__('sys').argv:
        return no_tc_fixture_main()
    if '--no_tc_audit' in __import__('sys').argv:
        return no_tc_audit_main()
    if '--projection_audit' in __import__('sys').argv:
        return projection_audit_main()
    if '--projection' in __import__('sys').argv:
        return projection_main()
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
