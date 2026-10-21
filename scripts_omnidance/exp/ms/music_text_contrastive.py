"""Raw native H sequence alignment with same-image music counterfactual InfoNCE."""
import argparse
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from music_text_adapter import (MusicToH3Text, load_music_text_checkpoint, save_music_text_checkpoint, length_loss,
                                masked_token_loss, convert_music_text_to_no_text_encoder)
from music_text_align import (read_split, load_targets, load_features, batch_tensors, evaluate,
                              sha256, write_new, parallel_file_sha256, checked_file_sha256)


@torch.no_grad()
def native_statistics(targets, train, qmax):
    sums = torch.zeros(5120, dtype=torch.float64)
    squares = torch.zeros_like(sums)
    n = 0
    position_sum = torch.zeros(qmax, 5120, dtype=torch.float64)
    position_count = torch.zeros(qmax, dtype=torch.float64)
    for sid in train:
        value = targets[sid]['tail'][:qmax].double()
        sums += value.sum(0)
        squares += value.square().sum(0)
        n += len(value)
        position_sum[:len(value)] += value
        position_count[:len(value)] += 1
    if n < 2:
        raise ValueError('Native statistics need at least two training tokens')
    mean = sums / n
    std = (squares / n - mean.square()).clamp_min(0).sqrt().clamp_min(0.1)
    position_mean = (position_sum / position_count.clamp_min(1)[:, None]).float()
    return mean.float(), std.float(), position_mean, float(squares.sum() / (n * 5120))


def score_samples(student, teacher, valid_mask, mean, std, limit=64, temperature=0.1, student_mask=None):
    if (student.ndim != 3 or student.shape != teacher.shape or student.shape[-1] != 5120
            or valid_mask.shape != student.shape[:2] or valid_mask.dtype != torch.bool):
        raise ValueError('Expected matching actual raw H sequences and valid token mask')
    if limit < 1 or temperature <= 0 or not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError('Invalid sequence score inputs')
    if student_mask is None:
        student_mask = valid_mask
    if student_mask.shape != student.shape[:2] or student_mask.dtype != torch.bool:
        raise ValueError('Student validity must use its own predicted lengths')
    scores = []
    for a, b, mask, own_mask in zip(student, teacher, valid_mask, student_mask):
        indices = mask.nonzero().flatten()
        own_indices = own_mask.nonzero().flatten()
        if not len(indices) or not len(own_indices):
            raise ValueError('Require nonempty teacher and student sequences')
        if len(indices) > limit:
            selected = torch.linspace(0, len(indices) - 1, limit, device=indices.device).round().long()
            indices = indices[selected]
        if len(own_indices) > limit:
            selected = torch.linspace(0, len(own_indices) - 1, limit, device=own_indices.device).round().long()
            own_indices = own_indices[selected]
        # Fixed train-only normalization is for the loss, never the deployed raw condition.
        x = F.normalize((a.index_select(0, own_indices).float() - mean) / std, dim=-1)
        y = F.normalize((b.index_select(0, indices).detach().float() - mean) / std, dim=-1)
        similarity = x @ y.T
        row = temperature * (torch.logsumexp(similarity / temperature, dim=-1) - math.log(len(indices)))
        col = temperature * (torch.logsumexp(similarity / temperature, dim=-2) - math.log(len(own_indices)))
        scores.append(0.5 * (row.mean() + col.mean()))
    return torch.stack(scores)


def counterfactual_loss(correct, wrong, zero, teacher, mask, mean, std):
    scores = torch.stack([score_samples(value, teacher, mask, mean, std) for value in (correct, wrong, zero)], dim=1)
    loss = F.cross_entropy(scores / 0.07, torch.zeros(len(scores), device=scores.device, dtype=torch.long))
    return loss, scores


def tie_aware_accuracy(scores):
    best = scores.max(1, keepdim=True).values
    ties = torch.isclose(scores, best, rtol=0, atol=1e-6)
    return ties[:, 0].float() / ties.sum(1)


def music_tensor(ids, features):
    lengths = [features[sid].shape[1] for sid in ids]
    value = torch.stack([F.pad(features[sid], (0, 0, 0, max(lengths) - n)) for sid, n in zip(ids, lengths)]).cuda().float()
    return value, torch.tensor(lengths, device='cuda')


def mismatched_ids(ids, all_ids, targets, offset=1, excluded=None):
    positions = {sid: i for i, sid in enumerate(all_ids)}
    result = []
    for index, sid in enumerate(ids):
        for shift in range(offset, offset + len(all_ids)):
            candidate = all_ids[(positions[sid] + shift) % len(all_ids)]
            if (candidate != sid and (excluded is None or candidate != excluded[index])
                    and targets[candidate]['metadata']['caption_sha256'] != targets[sid]['metadata']['caption_sha256']):
                result.append(candidate)
                break
        else:
            raise ValueError('No nonidentical caption/music candidate')
    return result


@torch.no_grad()
def dependence(model, ids, targets, features, mean, std, positional_mean, teacher_energy, zero_features, zero_lengths, batch_size):
    model.eval()
    rows = []
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        music, context, lengths, target, text_lengths, mask = batch_tensors(batch, targets, features, model.adapter.qmax)
        correct, _ = model.adapter(music, context, lengths, query_count=target.shape[1])
        wrong_ids = mismatched_ids(batch, ids, targets)
        wrong_music, wrong_lengths = music_tensor(wrong_ids, features)
        wrong, _ = model.adapter(wrong_music, context, wrong_lengths, query_count=target.shape[1])
        zero, _ = model.adapter(zero_features.expand(len(batch), -1, -1, -1), context,
                               zero_lengths.expand(len(batch)), query_count=target.shape[1])
        _, scores = counterfactual_loss(correct, wrong, zero, target, mask, mean, std)
        baseline = positional_mean[:target.shape[1]][None].expand(len(batch), -1, -1)
        baseline_score = score_samples(baseline, target, mask, mean, std)
        for i, sid in enumerate(batch):
            valid = mask[i]
            c, w, z, t = correct[i][valid], wrong[i][valid], zero[i][valid], target[i][valid]
            rows.append(dict(sample_id=sid, wrong_audio_id=wrong_ids[i], target_tokens=int(text_lengths[i]),
                    scores=scores[i].tolist(), tie_aware_correct_top1=float(tie_aware_accuracy(scores[i:i + 1])[0]),
                    correct_minus_wrong_score=float(scores[i, 0] - scores[i, 1]),
                    correct_minus_zero_score=float(scores[i, 0] - scores[i, 2]),
                    baseline_mean_score=float(baseline_score[i]),
                    raw_mse=float((c - t).square().mean()), wrong_raw_mse=float((w - t).square().mean()),
                    zero_raw_mse=float((z - t).square().mean()), position_mean_raw_mse=float((baseline[i][valid] - t).square().mean()),
                    correct_wrong_output_rms=float((c - w).square().mean().sqrt()),
                    student_teacher_energy_ratio=float(c.square().mean() / t.square().mean().clamp_min(1e-12))))
    keys = ['tie_aware_correct_top1', 'correct_minus_wrong_score', 'correct_minus_zero_score', 'raw_mse',
            'wrong_raw_mse', 'zero_raw_mse', 'position_mean_raw_mse', 'correct_wrong_output_rms', 'student_teacher_energy_ratio']
    macro = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    macro['chance_top1'] = 1 / 3
    macro['correct_beats_wrong_fraction'] = sum(row['correct_minus_wrong_score'] > 1e-6 for row in rows) / len(rows)
    macro['correct_beats_zero_fraction'] = sum(row['correct_minus_zero_score'] > 1e-6 for row in rows) / len(rows)
    return dict(macro=macro, cases=rows, candidates=len(rows), mean_baseline_music_choice_top1=1 / 3,
                same_context_and_loss_token_indices_for_all_branches=True,
                predicted_length_not_used_to_score=True, scope='Native raw H representation and music dependence, not generation adherence')


def accepted_checkpoint(directory):
    directory = Path(directory)
    report = json.loads((directory / 'finished.json').read_text())
    checkpoint = directory / 'last.pt'
    if (not report['complete'] or not report['musical_dependency_gate']
            or not report['complete_checkpoint_roundtrip_exact'] or not report['predicted_length_same']
            or sha256(checkpoint) != report['checkpoint_sha256']):
        raise ValueError('Native-H candidate has not passed completed musical-dependency/deployment gates')
    return checkpoint


def _state_hashes(model):
    import hashlib
    result = {}
    for name, value in model.state_dict().items():
        cpu = value.detach().cpu().contiguous()
        result[name] = hashlib.sha256(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    return result


def _gradient_scope(model):
    groups = {'backbone': [], 'adapter': []}
    missing, bad = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                bad.append(name + ': frozen parameter has gradient')
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        if not torch.isfinite(parameter.grad).all():
            bad.append(name)
        groups['backbone' if name.startswith('backbone.') else 'adapter'].append(
            (name, parameter.numel(), float(parameter.grad.detach().float().norm())))
    if missing or bad:
        raise ValueError(f'Invalid online gradient scope: missing={missing}, bad={bad}')
    report = {}
    for name, rows in groups.items():
        total = math.sqrt(sum(norm * norm for _, _, norm in rows))
        if (name == 'adapter' or not model.freeze_backbone) and total <= 0:
            raise ValueError('No nonzero gradient for ' + name)
        report[name] = dict(parameter_tensors=len(rows), parameter_elements=sum(n for _, n, _ in rows),
                            gradient_norm=total, zero_gradient_tensors=[key for key, _, norm in rows if norm == 0])
    return report


def _refresh_online_reservation():
    import os
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if gpu.isdigit():
        reservation = Path('/tmp/od_reserve_' + gpu)
        if reservation.is_file():
            reservation.touch()


def _online_batch(ids, bundle, qmax, context_free=False):
    wave = torch.stack([bundle['waveforms'][sid] for sid in ids]).cuda()
    context = None if context_free else torch.stack(
        [bundle['targets'][sid]['prefix_context'] for sid in ids]).cuda().float()
    tails = [bundle['targets'][sid]['tail'] for sid in ids]
    target = torch.stack([F.pad(value[:qmax], (0, 0, 0, qmax - min(len(value), qmax)))
                          for value in tails]).cuda().float()
    lengths = torch.tensor([len(value) for value in tails], device='cuda')
    teacher_mask = torch.arange(qmax, device='cuda')[None] < lengths[:, None]
    return wave, context, target, lengths, teacher_mask


def _online_scores(conditions, teacher, teacher_mask, mean, std):
    return torch.stack([score_samples(c['raw_tokens'], teacher, teacher_mask, mean, std,
                                     student_mask=c['valid_mask']) for c in conditions], dim=1)


def _online_conditions(model, ids, candidates, bundle, context, offset):
    if model.adapter.context_policy == 'music_only_global_bias' and context is not None:
        raise ValueError('noTC online conditions forbid cached or external context')
    targets = bundle['targets']
    first = mismatched_ids(ids, candidates, targets, offset=offset)
    second = mismatched_ids(ids, candidates, targets, offset=offset + 1, excluded=first)
    outputs = []
    for names in (ids, first, second):
        wave = torch.stack([bundle['waveforms'][sid] for sid in names]).cuda()
        outputs.append(model.student_condition(wave, context))
    return outputs, first, second


def _online_training_losses(model, wave, context, target, lengths, mask, ids, candidates,
                            bundle, mean, std, clip_weight, offset):
    if clip_weight == 0:
        correct = model.student_condition(wave, context)
        nce = None
    else:
        conditions, _, _ = _online_conditions(model, ids, candidates, bundle, context, offset)
        correct = conditions[0]
        scores = _online_scores(conditions, target, mask, mean, std)
        nce = F.cross_entropy(scores / 0.07, torch.zeros(len(ids), dtype=torch.long, device=scores.device))
    mse = masked_token_loss(correct['raw_tokens'], target, mask)['raw_mse']
    ll = length_loss(correct['log_length'], lengths, model.adapter.qmax)
    return mse, ll, nce


@torch.no_grad()
def _evaluate_online(model, ids, candidates, bundle, mean, std, position_mean, batch_size):
    model.eval()
    rows = []
    raw_sum, wrong_sum, zero_sum, teacher_sum, student_sum, count = 0., 0., 0., 0., 0., 0
    for start in range(0, len(ids), batch_size):
        _refresh_online_reservation()
        names = ids[start:start + batch_size]
        wave, context, target, lengths, mask = _online_batch(
            names, bundle, model.adapter.qmax, context_free=model.adapter.context_policy == 'music_only_global_bias')
        conditions, wrong_ids, second_ids = _online_conditions(model, names, candidates, bundle, context, 1)
        zero = model.student_condition(torch.zeros_like(wave), context)
        scores = _online_scores(conditions, target, mask, mean, std)
        zero_scores = _online_scores([zero], target, mask, mean, std)[:, 0]
        for i, sid in enumerate(names):
            c, w, z, t = (conditions[0]['raw_tokens'][i][mask[i]], conditions[1]['raw_tokens'][i][mask[i]],
                          zero['raw_tokens'][i][mask[i]], target[i][mask[i]])
            numel = t.numel()
            raw_sum += float((c - t).square().sum())
            wrong_sum += float((w - t).square().sum())
            zero_sum += float((z - t).square().sum())
            teacher_sum += float(t.square().sum())
            student_sum += float(c.square().sum())
            count += numel
            rows.append(dict(sample_id=sid, wrong_audio_id=wrong_ids[i], second_audio_id=second_ids[i],
                target_length=int(lengths[i]), predicted_lengths=[int(o['lengths'][i]) for o in conditions],
                zero_length=int(zero['lengths'][i]), scores=scores[i].tolist(), zero_score=float(zero_scores[i]),
                correct_minus_wrong_score=float(scores[i, 0] - scores[i, 1]),
                correct_minus_second_score=float(scores[i, 0] - scores[i, 2]),
                correct_minus_zero_score=float(scores[i, 0] - zero_scores[i]),
                tie_aware_top1=float(tie_aware_accuracy(scores[i:i + 1])[0]),
                raw_mse=float((c - t).square().mean()), wrong_raw_mse=float((w - t).square().mean()),
                zero_raw_mse=float((z - t).square().mean()),
                position_mean_mse=float((position_mean[:len(t)] - t).square().mean()),
                correct_wrong_output_rms=float((c - w).square().mean().sqrt()),
                length_absolute_error=abs(int(conditions[0]['lengths'][i]) - int(lengths[i])),
                length_log_mse=float((conditions[0]['log_length'][i].float() - lengths[i].float().log()).square()),
                supervised_tokens=int(mask[i].sum()), total_target_tokens=int(lengths[i]),
                truncated_supervision_tokens=max(0, int(lengths[i]) - model.adapter.qmax),
                length_unable_to_predict=int(lengths[i]) > model.adapter.qmax))
    macro = {key: sum(row[key] for row in rows) / len(rows) for key in (
        'correct_minus_wrong_score', 'correct_minus_second_score', 'correct_minus_zero_score',
        'tie_aware_top1', 'raw_mse', 'wrong_raw_mse', 'zero_raw_mse', 'position_mean_mse',
        'correct_wrong_output_rms', 'length_absolute_error', 'length_log_mse')}
    total_tokens = sum(row['total_target_tokens'] for row in rows)
    truncated_tokens = sum(row['truncated_supervision_tokens'] for row in rows)
    macro.update(total_target_tokens=total_tokens, supervised_tokens=total_tokens - truncated_tokens,
                 truncated_supervision_tokens=truncated_tokens,
                 truncated_supervision_fraction=truncated_tokens / total_tokens,
                 length_unable_to_predict_cases=sum(row['length_unable_to_predict'] for row in rows),
                 length_unable_to_predict_fraction=sum(row['length_unable_to_predict'] for row in rows) / len(rows))
    macro.update(correct_beats_wrong_fraction=sum(r['correct_minus_wrong_score'] > 1e-6 for r in rows) / len(rows),
                 correct_beats_second_fraction=sum(r['correct_minus_second_score'] > 1e-6 for r in rows) / len(rows),
                 raw_mse_tokenweighted=raw_sum / count, wrong_mse_tokenweighted=wrong_sum / count,
                 zero_mse_tokenweighted=zero_sum / count, student_teacher_energy_ratio=student_sum / max(teacher_sum, 1e-12))
    return dict(candidates=len(rows), macro=macro, cases=rows, student_uses_predicted_valid_interval=True,
                teacher_only_controls_supervision=True, silence_is_eval_only=True,
                raw_mse_scope='all_5120_raw_channels_of_min_true_target_length_qmax_tokens',
                length_label_scope='true_full_teacher_length_including_out_of_query_bound',
                query_max=model.adapter.qmax,
                scope='Representation diagnostics, not dance or video accuracy')


def _online_deployment_audit(model, checkpoint, wave, context, out):
    import os
    import subprocess
    import tempfile
    from music_text_adapter import load_music_text_checkpoint
    context_free = model.adapter.context_policy == 'music_only_global_bias'
    if context_free and context is not None:
        raise ValueError('noTC deployment audit forbids external/cached context')
    probe_context = None if context_free else context[:1]
    model.train()
    with torch.enable_grad():
        train_condition = model.student_condition(wave[:1], probe_context)
    if not train_condition['raw_tokens'].requires_grad or not train_condition['log_length'].requires_grad:
        raise ValueError('Training student condition lost its graph')
    expected = {key: train_condition[key].detach().cpu() for key in ('raw_tokens', 'log_length', 'lengths', 'valid_mask')}
    expected_tail = train_condition['tails'][0].detach().cpu()
    del train_condition
    model.eval()
    with torch.no_grad():
        actual = model.student_condition(wave[:1], probe_context)
        if not all(torch.equal(expected[key], actual[key].cpu()) for key in expected):
            raise ValueError('Train/eval student construction differs')
    del actual
    fresh = load_music_text_checkpoint(checkpoint, expected_backend='musicfm').cuda().eval()
    with torch.no_grad():
        actual = fresh.student_condition(wave[:1], probe_context)
        if not all(torch.equal(expected[key], actual[key].cpu()) for key in expected):
            raise ValueError('Full checkpoint changes online student condition')
    del actual, fresh
    model.cpu()
    torch.cuda.empty_cache()
    with tempfile.TemporaryDirectory(prefix='music_online_deploy_') as directory:
        inp, result = Path(directory) / 'input.pt', Path(directory) / 'result.pt'
        inputs = dict(waveform=wave[0].cpu())
        if not context_free:
            inputs['context'] = context[0].cpu()
        torch.save(inputs, inp)
        command = [__import__('sys').executable, str(Path(__file__).with_name('music_text_infer.py')),
                   '--checkpoint', str(checkpoint), '--checkpoint_sha256', sha256(checkpoint),
                   '--input', str(inp), '--out', str(result)]
        completed = subprocess.run(command, capture_output=True, text=True, env=os.environ.copy())
        write_new(out / 'independent_deployment_process.json', dict(command=command, returncode=completed.returncode,
                    stdout=completed.stdout, stderr=completed.stderr, input_keys=sorted(inputs),
                    context_policy=model.adapter.context_policy,
                    inference_source_sha256=sha256(Path(__file__).with_name('music_text_infer.py'))))
        if completed.returncode:
            raise RuntimeError('Independent deployment failed: ' + completed.stderr[-4000:])
        deployed = torch.load(result, map_location='cpu', weights_only=True)
    if (not torch.equal(expected_tail, deployed['tail'])
            or int(expected['lengths'][0]) != deployed['predicted_length']
            or deployed.get('context_policy', 'tc_prefix_context') != model.adapter.context_policy
            or deployed.get('checkpoint_sha256') != sha256(checkpoint)):
        raise ValueError('Independent inference changes actual deployed tail, length, checkpoint or context policy')
    # H3 prefix/tags/packed parity is audited separately in the H3 environment.
    probe = dict(waveform=wave[0].cpu(), tail=expected_tail, length=int(expected['lengths'][0]),
                 checkpoint_sha256=sha256(checkpoint))
    if not context_free:
        probe['context'] = context[0].cpu()
    write_new(out / 'deployment_probe.pt', probe, binary=True)
    return dict(train_eval_raw_length_mask_exact=True, complete_checkpoint_condition_exact=True,
                independent_inference_tail_exact=True, training_graph_present=True,
                context_policy=model.adapter.context_policy, cached_context_read=not context_free,
                h3_prefix_tags_packed_checked=False)


def _verify_online_initialization(init, bundle_init_sha256, bundle_sha256, pure_text=False):
    actual_sha256 = sha256(init)
    if pure_text and actual_sha256 != bundle_init_sha256:
        raise ValueError('Pure-text bundle init_sha256 must equal the actual complete initialization checkpoint SHA')
    if actual_sha256 == bundle_init_sha256:
        return actual_sha256, None
    finished_path = init.parent / 'finished.json'
    if init.name != 'last.pt' or not finished_path.is_file():
        raise ValueError('Changed online initialization requires the completed C300 parent last.pt')
    finished = json.loads(finished_path.read_text())
    protocol = finished.get('protocol', {})
    if (finished.get('complete') is not True or type(finished.get('steps')) is not int
            or finished['steps'] != 300 or protocol.get('steps') != 300
            or finished.get('checkpoint_sha256') != actual_sha256
            or protocol.get('init_sha256') != bundle_init_sha256
            or protocol.get('online_bundle_sha256') != bundle_sha256):
        raise ValueError('C300 initialization must bind its finished checkpoint SHA, original init and online bundle')
    return actual_sha256, dict(finished=str(finished_path), finished_sha256=sha256(finished_path),
                               checkpoint_sha256=actual_sha256, steps=300,
                               init_sha256=bundle_init_sha256, online_bundle_sha256=bundle_sha256)


def _load_output_space_manifest(path, checkpoint_sha256, bundle_sha256, train_ids, parent_finished_sha256):
    manifest = json.loads(path.read_text())
    finished_path = path.parent / 'finished.json'
    finished = json.loads(finished_path.read_text())
    variant = manifest.get('variant')
    if (manifest.get('complete') is not True or manifest.get('format_version') != 1
            or manifest.get('diagnostic_only') is not True
            or manifest.get('kind') != 'projected_teacher_not_music_prediction'
            or manifest.get('teacher_length_for_oracle_only') is not True
            or manifest.get('changes_student_deployment') is not False
            or manifest.get('checkpoint_sha256') != checkpoint_sha256
            or manifest.get('bundle_sha256') != bundle_sha256
            or finished.get('complete') is not True or finished.get('optimizer_updates') != 0
            or finished.get('checkpoint_sha256') != checkpoint_sha256
            or finished.get('bundle_sha256') != bundle_sha256
            or finished.get('checkpoint_summary_sha256') != parent_finished_sha256
            or finished.get('pca_uses_train_only') is not True
            or variant not in ('w256', 'ln255', 'pca256', 'pca256n', 'pca512n')
            or str(path.resolve()) not in {str(Path(item).resolve()) for item in finished.get('manifests', [])}):
        raise ValueError('Output-space manifest must bind a completed read-only train-only diagnostic to this checkpoint/bundle')
    rows = [item for item in finished.get('variants', []) if item.get('variant') == variant]
    if len(rows) != 1 or Path(rows[0]['oracle_manifest']).resolve() != path.resolve():
        raise ValueError('Output space must be the unique declared completed diagnostic variant')
    space_metadata = rows[0].get('space_metadata')
    if not isinstance(space_metadata, dict):
        raise ValueError('Output space requires complete source metadata')
    if variant.startswith('pca'):
        sampling = space_metadata.get('sampling', [])
        if (space_metadata.get('validation_used') is not False
                or space_metadata.get('prefix_and_padding_used') is not False
                or space_metadata.get('train_samples') != len(train_ids)
                or [item.get('sample_id') for item in sampling] != train_ids):
            raise ValueError('PCA output-space fitting must use exactly the declared full train-only token pool')
    elif (space_metadata.get('svd_dtype') != 'float64'
          or type(space_metadata.get('numerical_rank')) is not int):
        raise ValueError('Checkpoint-derived output space requires its exact FP64 SVD metadata')
    if 'space_metadata' in manifest and manifest['space_metadata'] != space_metadata:
        raise ValueError('Manifest output-space metadata differs from completed diagnostic')
    if 'train_ids' in manifest and manifest['train_ids'] != train_ids:
        raise ValueError('Manifest train pool differs from online bundle')
    numeric = Path(manifest['output_space'])
    source_sha256 = sha256(numeric)
    if source_sha256 != manifest.get('output_space_sha256'):
        raise ValueError('Output-space numerical artifact SHA mismatch')
    state = torch.load(numeric, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or set(state) != {'center', 'basis'}:
        raise ValueError('Output-space numerical artifact must contain only center and basis')
    rank = space_metadata.get('rank', space_metadata.get('numerical_rank'))
    if (type(rank) is not int or not isinstance(state['basis'], torch.Tensor)
            or state['basis'].ndim != 2 or state['basis'].shape[1] != rank):
        raise ValueError('Output-space basis rank differs from completed diagnostic metadata')
    source = dict(manifest=str(path), manifest_sha256=sha256(path), finished=str(finished_path),
                  finished_sha256=sha256(finished_path), artifact=str(numeric), output_space_sha256=source_sha256,
                  checkpoint_sha256=checkpoint_sha256, checkpoint_summary_sha256=parent_finished_sha256,
                  online_bundle_sha256=bundle_sha256, variant=variant, space_metadata=space_metadata)
    return state, source


PURE_TEXT_PROTOCOL = 'musicfm_pure_text_h3_raw_tokens_v1'


def _validate_pure_text_bundle(bundle, train, val):
    metadata = bundle['metadata']
    pure_text = metadata.get('teacher_kind') == 'pure_text_h3_raw_tokens'
    if not pure_text:
        if metadata.get('teacher_kind') is not None:
            raise ValueError('Unsupported online teacher_kind')
        return False
    if (type(metadata.get('format_version')) is not int or metadata['format_version'] != 2
            or metadata.get('protocol') != PURE_TEXT_PROTOCOL
            or metadata.get('teacher_image_conditioned') is not False
            or type(metadata.get('teacher_prefix_length')) is not int or metadata['teacher_prefix_length'] != 0
            or metadata.get('student_feature_cache_used') is not False
            or metadata.get('initialization_binding') != 'external_pure_text_init_binding_manifest'
            or 'init_sha256' in metadata):
        raise ValueError('Pure-text online bundle requires its explicit external-init, no-image/no-prefix protocol')
    for sid in train + val:
        target = bundle['targets'].get(sid)
        if not isinstance(target, dict) or set(target) != {'metadata', 'tail', 'source_token_ids', 'teacher_length'}:
            raise ValueError('Pure-text target requires exactly metadata/tail/source_token_ids/teacher_length: ' + sid)
        tail, token_ids, length, row = (target[key] for key in ('tail', 'source_token_ids', 'teacher_length', 'metadata'))
        if (type(length) is not int or length < 1 or not isinstance(tail, torch.Tensor)
                or tail.shape != (length, 5120) or tail.dtype != torch.bfloat16 or not torch.isfinite(tail).all()
                or not isinstance(token_ids, torch.Tensor) or token_ids.dtype != torch.long
                or token_ids.shape != (length,) or (token_ids < 0).any()
                or not isinstance(row, dict) or row.get('kind') != 'pure_text_h3_raw_tokens'
                or row.get('text_only') is not True or row.get('image_conditioned') is not False
                or type(row.get('prefix_length')) is not int or row['prefix_length'] != 0
                or 'prefix_context' in row):
            raise ValueError('Pure-text target must be finite full raw text-only H3 tokens without context: ' + sid)
    return True


def _verify_pure_init_provenance(path, init, bundle_sha256, metadata):
    if path is None:
        raise ValueError('Pure-text online training requires --init_provenance; embedded init SHA is not accepted')
    record = json.loads(Path(path).read_text())
    init_sha256 = sha256(init)
    if (not isinstance(record, dict) or type(record.get('format_version')) is not int
            or record['format_version'] != 1 or record.get('kind') != 'pure_text_init_binding'
            or record.get('complete') is not True or record.get('bundle_sha256') != bundle_sha256
            or record.get('init_sha256') != init_sha256 or record.get('split_sha256') != metadata['split_sha256']
            or type(record.get('query_max')) is not int or record['query_max'] != metadata['query_max']
            or record.get('teacher_kind') != 'pure_text_h3_raw_tokens'
            or record.get('teacher_image_conditioned') is not False
            or record.get('context_policy') != 'music_only_global_bias'
            or record.get('head_normalization') not in ('identity', 'layer_norm')
            or record.get('student_inputs') != 'waveform_only'
            or record.get('no_fixed_output_projection') is not True
            or record.get('bundle_copied') is not False
            or type(record.get('optimizer_updates')) is not int or record['optimizer_updates'] != 0
            or type(record.get('init_report_path')) is not str):
        raise ValueError('Pure-text initializer provenance does not bind the actual bundle/init and waveform-only free-head scope')
    report_path = Path(init).with_suffix('.json')
    if Path(record['init_report_path']).resolve() != report_path.resolve():
        raise ValueError('Pure-text initializer report must be the actual init checkpoint sibling JSON')
    if sha256(report_path) != record.get('init_report_sha256'):
        raise ValueError('Pure-text initializer report SHA mismatch')
    report = json.loads(report_path.read_text())
    if (not isinstance(report, dict) or report.get('complete') is not True
            or report.get('init_sha256') != init_sha256 or report.get('bundle_sha256') != bundle_sha256
            or type(report.get('optimizer_updates')) is not int or report['optimizer_updates'] != 0
            or type(report.get('backward_calls')) is not int or report['backward_calls'] != 0
            or report.get('query_max') != metadata['query_max']
            or report.get('context_policy') != record['context_policy']
            or report.get('head_normalization') != record['head_normalization']
            or report.get('output_space') != {'mode': 'none'}
            or not isinstance(report.get('checkpoint_metadata'), dict)
            or not isinstance(report.get('provenance'), dict)
            or report['provenance'].get('kind') != record.get('initialization_kind')):
        raise ValueError('Pure-text initializer report scope/checkpoint binding mismatch')
    source = dict(path=str(path), sha256=sha256(path), manifest=record,
                  actual_init_sha256=init_sha256, actual_bundle_sha256=bundle_sha256,
                  init_report_path=str(report_path), init_report_sha256=sha256(report_path), init_report=report)
    return init_sha256, source


def _validate_pure_init_model(model, source, metadata):
    report, manifest = source['init_report'], source['manifest']
    checkpoint_metadata = getattr(model, 'checkpoint_metadata', None)
    if (not isinstance(model, MusicToH3Text) or not isinstance(checkpoint_metadata, dict)
            or checkpoint_metadata != report['checkpoint_metadata']
            or checkpoint_metadata.get('state_scope') != 'music_to_h3_text'
            or checkpoint_metadata.get('format_version') != 3
            or model.adapter.qmax != metadata['query_max'] or model.adapter.width != report.get('width')
            or model.adapter.head_normalization != manifest['head_normalization']
            or model.adapter.context_policy != manifest['context_policy']
            or model.adapter.output_space_metadata() != {'mode': 'none'}):
        raise ValueError('Pure-text complete initializer actual metadata disagrees with its strict provenance')
    return dict(actual_checkpoint_metadata=checkpoint_metadata,
                actual_checkpoint_state_keys=sorted(model.state_dict()),
                cached_context_read=False, text_encoder_called=False, student_inputs='waveform_only')


def online_main():
    parser = argparse.ArgumentParser(description='Online MusicFM dense backbone + complete raw H adapter')
    parser.add_argument('--online', action='store_true', required=True)
    parser.add_argument('--online_bundle', type=Path, required=True)
    parser.add_argument('--init_provenance', type=Path,
                        help='Required strict external bundle/init/report SHA binding for pure-text V2 bundles')
    parser.add_argument('--output_space_manifest', type=Path,
                        help='Completed checkpoint-W or train-only PCA manifest bound to this init checkpoint and online bundle')
    parser.add_argument('--init', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--backbone_train', action='store_true')
    parser.add_argument('--no_text_encoder', action='store_true',
                        help='Use SHA-bound pure-text noTC initialization, or convert verified legacy C300; never read cached TC context')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--microbatch', type=int, default=2)
    parser.add_argument('--backbone_lr', type=float, default=1e-5)
    parser.add_argument('--adapter_lr', type=float, default=1e-4)
    parser.add_argument('--clip_weight', type=float, default=1.)
    parser.add_argument('--length_weight', type=float, default=1.)
    parser.add_argument('--small_fit', type=int, default=0)
    parser.add_argument('--validation_count', type=int, default=0)
    args = parser.parse_args()
    if (args.out.exists() or not args.out.parent.is_dir() or args.steps < 1 or args.microbatch < 1
            or args.batch_size < args.microbatch or args.batch_size % args.microbatch
            or args.small_fit < 0 or args.validation_count < 0
            or args.small_fit in (1, 2) or args.validation_count in (1, 2)
            or any(not math.isfinite(x) or x <= 0 for x in (args.backbone_lr, args.adapter_lr, args.length_weight))
            or not math.isfinite(args.clip_weight) or args.clip_weight < 0):
        parser.error('Require new output, valid counts and finite loss/optimizer values')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    random.seed(42)
    report = json.loads(args.online_bundle.with_suffix('.json').read_text())
    bundle_hash = sha256(args.online_bundle)
    if not report['complete'] or report['sha256'] != bundle_hash:
        raise ValueError('Online waveform/teacher bundle SHA mismatch')
    bundle = torch.load(args.online_bundle, map_location='cpu', weights_only=True)
    expected_bundle_keys = {'metadata', 'targets', 'waveforms', 'mean', 'std', 'position_mean', 'teacher_energy', 'protocols'}
    has_pure_artifacts = isinstance(bundle.get('metadata'), dict) and bundle['metadata'].get('teacher_kind') == 'pure_text_h3_raw_tokens'
    if has_pure_artifacts:
        expected_bundle_keys.add('artifacts')
        artifacts = bundle.get('artifacts')
        if not isinstance(artifacts, dict) or set(artifacts) != {'teacher_release', 'caps'}:
            raise ValueError('Pure bundle must bind exactly its actual teacher release and captions')
        for key, metadata_key in (('teacher_release', 'teacher_release_report'), ('caps', 'caps')):
            artifact = artifacts[key]
            if (not isinstance(artifact, dict) or set(artifact) != {'path', 'sha256', 'content'}
                    or {k: artifact[k] for k in ('path', 'sha256')} != bundle['metadata'].get(metadata_key)
                    or sha256(Path(artifact['path'])) != artifact['sha256']
                    or json.loads(Path(artifact['path']).read_text()) != artifact['content']):
                raise ValueError('Pure bundle source artifact changed: ' + key)
    if set(bundle) != expected_bundle_keys:
        raise ValueError('Online bundle keys must match its exact teacher protocol; student features are forbidden')
    train, val = read_split(args.split)
    metadata = bundle['metadata']
    pure_text = _validate_pure_text_bundle(bundle, train, val)
    init_provenance, init_model_audit = None, None
    if pure_text:
        run_init_sha256, init_provenance = _verify_pure_init_provenance(
            args.init_provenance, args.init, bundle_hash, metadata)
        parent_c300 = None
    else:
        if args.init_provenance is not None:
            raise ValueError('--init_provenance is restricted to the pure-text V2 protocol; legacy init gates remain unchanged')
        run_init_sha256, parent_c300 = _verify_online_initialization(args.init, metadata['init_sha256'], bundle_hash)
    if (metadata['format_version'] != (2 if pure_text else 1) or metadata['split_sha256'] != sha256(args.split)
            or metadata['train_ids'] != train or metadata['val_ids'] != val
            or metadata['student_feature_cache_used'] is not False
            or set(bundle['targets']) != set(train + val) or set(bundle['waveforms']) != set(train + val)):
        raise ValueError('Online bundle/initialization/split mismatch')
    if pure_text and not args.no_text_encoder:
        raise ValueError('Pure-text teacher bundles require explicit --no_text_encoder')
    if args.no_text_encoder and not pure_text and parent_c300 is None:
        raise ValueError('--no_text_encoder legacy mode requires the verified completed C300 common initialization')
    if pure_text and args.output_space_manifest:
        raise ValueError('Pure-text direct initialization must train free raw-output weights, not a fixed output space')
    space_state, space_source = None, None
    if args.output_space_manifest:
        if parent_c300 is None:
            raise ValueError('Output-space student comparisons must start from the verified completed C300 checkpoint')
        space_state, space_source = _load_output_space_manifest(
            args.output_space_manifest, run_init_sha256, bundle_hash, train, parent_c300['finished_sha256'])
    from music_clip_bridge import load_audio
    for sid in (train[0], train[1], val[0]):
        if not torch.equal(bundle['waveforms'][sid], load_audio(sid)[0]):
            raise ValueError('Cached waveform differs from deployment loader: ' + sid)
    for sid, value in bundle['waveforms'].items():
        if value.dtype != torch.float32 or value.shape != (2, 162000) or not torch.isfinite(value).all():
            raise ValueError('Invalid online waveform: ' + sid)
    mean, std, position_mean = (bundle[key].cuda() for key in ('mean', 'std', 'position_mean'))
    teacher_energy = float(bundle['teacher_energy'])
    if (mean.shape != (5120,) or std.shape != (5120,) or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all() or (std <= 0).any() or not math.isfinite(teacher_energy) or teacher_energy <= 0):
        raise ValueError('Invalid fixed train statistics')
    model = load_music_text_checkpoint(args.init, expected_backend='musicfm').cuda()
    if not isinstance(model, MusicToH3Text):
        raise ValueError('Online initialization requires a complete backbone and adapter checkpoint')
    if pure_text:
        init_model_audit = _validate_pure_init_model(model, init_provenance, metadata)
    model.configure_online_training(args.backbone_train)
    qmax = model.adapter.qmax
    if position_mean.shape != (qmax, 5120) or max(len(bundle['targets'][sid]['tail']) for sid in train) != qmax:
        raise ValueError('Query bound must come from full train only')
    context_conversion = None
    if args.no_text_encoder:
        if pure_text:
            context_conversion = dict(context_policy='music_only_global_bias', conversion_performed=False,
                already_music_only_global_bias=True, source_checkpoint_sha256=run_init_sha256,
                source_state_sha256=_state_hashes(model), query_max=qmax, adapter_width=model.adapter.width,
                head_normalization=model.adapter.head_normalization, output_space=model.adapter.output_space_metadata(),
                train_target_max=max(len(bundle['targets'][sid]['tail']) for sid in train),
                new_init_statistics=dict(
                    raw_output_weight_rms=float(model.adapter.raw_output.weight.detach().square().mean().sqrt()),
                    raw_output_bias_mean=float(model.adapter.raw_output.bias.detach().mean()),
                    raw_output_bias_rms=float(model.adapter.raw_output.bias.detach().square().mean().sqrt()),
                    query_position_rms=float(model.adapter.query_positions.weight.detach().square().mean().sqrt()),
                    global_context_bias_rms=float(model.adapter.global_context_bias.detach().square().mean().sqrt()),
                    train_teacher_energy=teacher_energy,
                    source_calibration=init_provenance['init_report'].get('initialization_statistics',
                                                                         init_provenance['init_report'].get('calibration'))),
                cached_context_read=False, text_encoder_called=False,
                all_remaining_adapter_parameters_trainable=all(p.requires_grad for p in model.adapter.parameters()))
        else:
            model.eval()
            with torch.no_grad():
                features, feature_lengths = model.dense_features(bundle['waveforms'][val[0]][None].cuda())
            context_conversion = convert_music_text_to_no_text_encoder(model, features, feature_lengths)
            context_conversion.update(source_checkpoint_sha256=run_init_sha256, probe_sample_id=val[0])
            del features, feature_lengths
    elif model.adapter.context_policy != 'tc_prefix_context':
        raise ValueError('Music-only checkpoints require an explicit noTC training protocol; legacy path must not supply context')
    context_free = model.adapter.context_policy == 'music_only_global_bias'
    projection_initial_state = None
    if space_state is not None:
        if model.adapter.output_basis is not None:
            raise ValueError('C300 common initialization must not already have an output-space projection')
        before_projection = _state_hashes(model)
        probe_id = val[0]
        model.eval()
        with torch.no_grad():
            probe_wave = bundle['waveforms'][probe_id][None].cuda()
            probe_context = None if context_free else bundle['targets'][probe_id]['prefix_context'][None].cuda().float()
            features, feature_lengths = model.dense_features(probe_wave)
            original_raw, original_length = model.adapter(features, probe_context, feature_lengths)
            model.adapter.set_output_space(space_state['center'], space_state['basis'],
                                           space_source['output_space_sha256'])
            projected_raw, projected_length = model.adapter(features, probe_context, feature_lengths)
        after_projection = _state_hashes(model)
        added = sorted(set(after_projection) - set(before_projection))
        changed = [name for name in before_projection if before_projection[name] != after_projection.get(name)]
        if (changed or added != ['adapter.output_basis', 'adapter.output_center']
                or not torch.equal(original_length, projected_length)):
            raise ValueError('Installing a fixed output space must only add buffers and leave all initial parameters/lengths unchanged')
        delta = projected_raw - original_raw
        projection_initial_state = dict(probe_sample_id=probe_id, probe_queries=qmax,
            added_persistent_buffers=added, changed_existing_state=[], optimizer_updates=0,
            raw_output_changed_before_training=not torch.equal(original_raw, projected_raw),
            raw_projection_delta_rms=float(delta.square().mean().sqrt()),
            raw_projection_relative_rms=float(delta.norm() / original_raw.norm().clamp_min(1e-12)),
            log_length_unchanged=True, width=model.adapter.width,
            affine_difference_rank_upper_bound=min(model.adapter.width - 1, model.adapter.output_basis.shape[1]),
            full_fixed_basis_reachability_claim=False,
            limitation='Trainable output_norm LayerNorm imposes width-1 affine envelope; nonlinear normalized queries are further constrained')
        del features, feature_lengths, original_raw, original_length, projected_raw, projected_length, delta, probe_wave, probe_context
    if args.small_fit:
        train = train[:args.small_fit]
        val = val[:min(len(val), args.small_fit)]
    if args.validation_count:
        val = val[:args.validation_count]
    args.out.mkdir()
    protocol = dict(init=str(args.init), init_sha256=run_init_sha256, run_init_sha256=run_init_sha256,
        bundle_init_sha256=metadata.get('init_sha256'), parent_c300=parent_c300,
        output_space_source=space_source, output_space=model.adapter.output_space_metadata(),
        projection_initial_state=projection_initial_state, optimizer_state='new_AdamW_not_resumed',
        adapter_width=model.adapter.width, adapter_heads=model.adapter.heads,
        no_text_encoder=args.no_text_encoder, context_policy=model.adapter.context_policy,
        context_conversion=context_conversion, student_cached_context_read=not context_free,
        student_inputs='music_waveform_only_global_trainable_bias' if context_free else 'music_waveform_and_cached_tc_prefix_context',
        text_encoder_called=False, teacher_hidden_and_length_for_supervision_only=True,
        student_image_condition=False, reference_image_scope='downstream_H3_visual_VAE_only',
        online_bundle_sha256=bundle_hash,
        split_sha256=sha256(args.split), train_ids=train, validation_ids=val, steps=args.steps,
        batch_size=args.batch_size, microbatch=args.microbatch, accumulation=args.batch_size // args.microbatch,
        backbone_train=args.backbone_train, backbone_lr=args.backbone_lr, adapter_lr=args.adapter_lr,
        clip_weight=args.clip_weight, length_weight=args.length_weight, seed=42,
        shared_student_condition=True, student_uses_ground_truth_length=False,
        online_waveform_backbone_all_three_branches=bool(args.clip_weight), student_feature_cache_used=False,
        training_student_forward_branches=3 if args.clip_weight else 1,
        infonce_computed=bool(args.clip_weight),
        negatives='two_different_ids_and_caption_hashes_not_verified_distinct_source_songs' if args.clip_weight else 'not_computed_paired_only',
        backbone_eval_behavior_with_parameter_gradients=args.backbone_train,
        dtype='float32_no_cached_feature_roundtrip', no_dit_video_loss=True,
        source_sha256=sha256(Path(__file__)), adapter_source_sha256=sha256(Path(__file__).with_name('music_text_adapter.py')),
        loss='rawMSE/train_teacher_energy + length_weight*log_length_MSE + clip_weight*min(step/100,1)*same_context_sequence_NCE',
        sampling='independent_up_to64_student_predicted_valid_tokens_and_teacher_valid_tokens',
        full_train_statistics_reused=True, small_fit=args.small_fit)
    if pure_text:
        protocol.update(protocol=PURE_TEXT_PROTOCOL, teacher_kind='pure_text_h3_raw_tokens',
                        teacher_scope='pure_text_h3_raw_tokens', adapter_head_normalization=model.adapter.head_normalization,
                        init_provenance=init_provenance, init_model_audit=init_model_audit,
                        query_max=qmax, query_bound_source='full_pure_text_train_target_max',
                        validation_long_targets='truncate_raw_supervision_explicitly_report_full_length_log_mse',
                        initialization_statistics=context_conversion['new_init_statistics'])
    if not args.clip_weight:
        protocol.update(loss='rawMSE/train_teacher_energy + length_weight*log_length_MSE',
                        sampling='paired_raw_tokens_only_no_sequence_score_or_negative_sampling')
    write_new(args.out / 'plan.json', protocol)
    parameters = dict(model.named_parameters())
    before = _state_hashes(model)
    initial = _evaluate_online(model, val, val, bundle, mean, std, position_mean, args.microbatch)
    write_new(args.out / 'initial_validation.json', initial)
    groups = [dict(params=list(model.adapter.parameters()), lr=args.adapter_lr, name='adapter')]
    if args.backbone_train:
        groups.insert(0, dict(params=[p for p in model.backbone.parameters() if p.requires_grad],
                             lr=args.backbone_lr, name='backbone'))
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    order, cursor, items_seen = [], 0, 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    first_gradient = None
    history = []
    for step in range(1, args.steps + 1):
        _refresh_online_reservation()
        ids = []
        while len(ids) < args.batch_size:
            if cursor >= len(order):
                order, cursor = random.sample(train, len(train)), 0
            take = min(args.batch_size - len(ids), len(order) - cursor)
            ids += order[cursor:cursor + take]
            cursor += take
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = dict(loss=0., raw_mse=0., length=0., infonce=0. if args.clip_weight else None,
                      infonce_computed=bool(args.clip_weight))
        batch_tokens = sum(len(bundle['targets'][sid]['tail']) for sid in ids)
        for start in range(0, len(ids), args.microbatch):
            names = ids[start:start + args.microbatch]
            wave, context, target, lengths, mask = _online_batch(names, bundle, qmax, context_free=context_free)
            mse, ll, nce = _online_training_losses(
                model, wave, context, target, lengths, mask, names, train, bundle, mean, std,
                args.clip_weight, step % max(1, len(train) - 1) + 1)
            weight = len(names) / len(ids)
            token_weight = int(mask.sum()) / batch_tokens
            loss = mse / teacher_energy * token_weight + weight * args.length_weight * ll
            if nce is not None:
                loss = mse / teacher_energy * token_weight + weight * (
                    args.length_weight * ll + args.clip_weight * min(1., step / 100) * nce)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite online training loss')
            loss.backward()
            totals['loss'] += float(loss.detach())
            totals['raw_mse'] += float(mse.detach()) * token_weight
            totals['length'] += float(ll.detach()) * weight
            if nce is not None:
                totals['infonce'] += float(nce.detach()) * weight
            del loss, mse, ll, nce
        if first_gradient is None:
            first_gradient = _gradient_scope(model)
            write_new(args.out / 'gradient_step1.json', first_gradient)
        elif any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Nonfinite later online gradient')
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step()
        items_seen += len(ids)
        if step == 1 or step % 10 == 0 or step == args.steps:
            record = dict(step=step, items_seen=items_seen, seconds=time.monotonic() - started, **totals,
                          cuda_peak_bytes=torch.cuda.max_memory_allocated())
            history.append(record)
            print(json.dumps(dict(event='online_train', **record)), flush=True)
    training_seconds = time.monotonic() - started
    after = _state_hashes(model)
    changed = [name for name in before if before[name] != after[name]]
    frozen_changed = [name for name in changed if name not in parameters or not parameters[name].requires_grad]
    backbone_changed = [name for name in changed if name.startswith('backbone.')]
    adapter_changed = [name for name in changed if name.startswith('adapter.')]
    if frozen_changed or not adapter_changed or bool(backbone_changed) != args.backbone_train:
        raise ValueError(f'Wrong actual weight/buffer update scope: frozen={frozen_changed}')
    weight_audit = dict(actual_changed_parameters=changed, backbone_changed_tensors=len(backbone_changed),
        adapter_changed_tensors=len(adapter_changed), frozen_parameters_and_buffers_unchanged=True,
        before_sha256=before, after_sha256=after, gradient_step1=first_gradient)
    write_new(args.out / 'weight_update_audit.json', weight_audit)
    final = _evaluate_online(model, val, val, bundle, mean, std, position_mean, args.microbatch)
    train_probe = _evaluate_online(model, train[:min(8, len(train))], train, bundle, mean, std, position_mean, args.microbatch)
    write_new(args.out / 'final_validation.json', final)
    write_new(args.out / 'final_train_probe.json', train_probe)
    checkpoint = args.out / 'last.pt'
    model.eval()
    save_music_text_checkpoint(checkpoint, model)
    wave, context, _, _, _ = _online_batch(val[:1], bundle, qmax, context_free=context_free)
    del optimizer
    deployment = _online_deployment_audit(model, checkpoint, wave, context, args.out)
    summary = dict(complete=True, protocol=protocol, steps=args.steps, actual_items_seen=items_seen,
        initial=initial['macro'], final=final['macro'], train_probe=train_probe['macro'],
        checkpoint_sha256=sha256(checkpoint), deployment=deployment,
        actual_backbone_changed_tensors=len(backbone_changed), actual_adapter_changed_tensors=len(adapter_changed),
        frozen_parameters_and_buffers_unchanged=True, training_seconds=training_seconds,
        cuda_peak_bytes=torch.cuda.max_memory_allocated(), history=history,
        video_generation_required=True, quality_metrics_do_not_block_diagnostic_inference=True,
        h3_condition_layout_audit_pending=True,
        scope='Online training and raw-condition deployment audited; video and H3 layout not yet accepted')
    write_new(args.out / 'finished.json', summary)
    print(json.dumps(dict(event='online_finished', out=str(args.out), **summary)), flush=True)


# Direct-token training is isolated from the legacy sequence/counterfactual paths.
DIRECT_CLIP_PROTOCOL = 'musicfm_direct_tokens_cached_prefix_v1'


def _direct_sample_statistics(raw, mask):
    if (raw.ndim != 3 or mask.shape != raw.shape[:2] or mask.dtype != torch.bool
            or not torch.isfinite(raw).all() or not mask.any(1).all()):
        raise ValueError('Direct loss requires finite tokens and a nonempty actual mask per sample')
    count = mask.sum(1).float()
    mean = raw.float().masked_fill(~mask[..., None], 0).sum(1) / count[:, None]
    centered = (raw.float() - mean[:, None]).masked_fill(~mask[..., None], 0)
    rms = (centered.square().sum((1, 2)) / (count * raw.shape[-1])).sqrt()
    return mean, rms


def _direct_moment_terms(student_mean, student_rms, teacher_mean, teacher_rms, std, eps=1e-6):
    location = ((student_mean - teacher_mean.detach()) / std).square().mean(-1)
    scale = ((student_rms + eps) / (teacher_rms.detach() + eps)).log().square()
    return location, scale


def _direct_clip_scores(student_mean, teacher_mean, mean, std, groups):
    student = F.normalize((student_mean.float() - mean) / std, dim=-1)
    teacher = F.normalize((teacher_mean.detach().float() - mean) / std, dim=-1)
    logits = student @ teacher.T / 0.07
    # Other rows of the same song are not negatives; the paired diagonal remains.
    group = torch.tensor(groups, device=logits.device, dtype=torch.long)
    excluded = group[:, None].eq(group[None, :])
    excluded.fill_diagonal_(False)
    return logits.masked_fill(excluded, float('-inf')), student, teacher


def _direct_losses(student_mean, student_rms, teacher_mean, teacher_rms, mean, std, groups, moment_weight):
    logits, student, teacher = _direct_clip_scores(student_mean, teacher_mean, mean, std, groups)
    labels = torch.arange(len(logits), device=logits.device)
    clip = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    location, scale = _direct_moment_terms(student_mean, student_rms, teacher_mean, teacher_rms, std)
    moment = (location + scale).mean()
    loss = clip + moment_weight * moment
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite direct CLIP/moments loss')
    return loss, dict(clip=clip, moment=moment, location=location.mean(), scale=scale.mean()), student, teacher


def _direct_gather(value, differentiable=False):
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return value
    size = torch.tensor([len(value)], device=value.device, dtype=torch.long)
    sizes = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, size)
    sizes = [int(item.item()) for item in sizes]
    maximum = max(sizes)
    padding = value.new_zeros((maximum - len(value), *value.shape[1:]))
    padded = torch.cat((value, padding), dim=0)
    if differentiable:
        from torch.distributed.nn.functional import all_gather
        gathered = all_gather(padded)
    else:
        gathered = [torch.empty_like(padded) for _ in sizes]
        dist.all_gather(gathered, padded)
    return torch.cat([item[:size] for item, size in zip(gathered, sizes)], dim=0)


def _direct_global_loss(raw, mask, teacher_mean, teacher_rms, groups, mean, std, moment_weight,
                        actual_rows=None):
    student_mean, student_rms = _direct_sample_statistics(raw, mask)
    if actual_rows is not None:
        student_mean, student_rms = student_mean[actual_rows], student_rms[actual_rows]
    student_mean = _direct_gather(student_mean, differentiable=True)
    student_rms = _direct_gather(student_rms, differentiable=True)
    teacher_mean = _direct_gather(teacher_mean)
    teacher_rms = _direct_gather(teacher_rms)
    group_tensor = torch.tensor(groups, device=raw.device, dtype=torch.long)
    all_groups = _direct_gather(group_tensor).tolist()
    # Every rank computes the same global loss. Autograd all_gather SUMs its
    # gradients across ranks, exactly cancelling DDP's parameter-gradient mean.
    return _direct_losses(student_mean, student_rms, teacher_mean, teacher_rms,
                          mean, std, all_groups, moment_weight)


def _direct_epoch_batches(ids, local_batch, world, seed, epoch):
    order = list(ids)
    random.Random(seed + epoch).shuffle(order)
    return [order[start:start + local_batch * world]
            for start in range(0, len(order), local_batch * world)]


def _direct_rank_ids(global_ids, rank, world):
    base, remainder = divmod(len(global_ids), world)
    start = rank * base + min(rank, remainder)
    return global_ids[start:start + base + (rank < remainder)]


def _direct_load_manifest(path, verify_sources=True):
    manifest = path if isinstance(path, dict) else json.loads(Path(path).read_text())
    if not isinstance(manifest, dict) or set(manifest) != {'metadata', 'rows', 'statistics'}:
        raise ValueError('Direct manifest requires metadata/rows/statistics only')
    metadata, rows, statistics = (manifest[key] for key in ('metadata', 'rows', 'statistics'))
    if (not isinstance(metadata, dict) or metadata.get('protocol') != DIRECT_CLIP_PROTOCOL
            or metadata.get('complete') is not True or not isinstance(rows, dict)
            or not isinstance(statistics, dict)):
        raise ValueError('Direct manifest is not a completed direct-token release')
    train, val = metadata.get('train_ids'), metadata.get('val_ids')
    if (not isinstance(train, list) or not isinstance(val, list) or len(train) != 6000 or len(val) != 500
            or any(type(sid) is not str or not sid for sid in train + val)
            or len(set(train + val)) != 6500 or set(rows) != set(train + val)):
        raise ValueError('Require the locked 6000/500 disjoint unique sample split')
    mean, std = (torch.tensor(statistics.get(key, []), dtype=torch.float32) for key in ('mean', 'std'))
    energy, token_count = statistics.get('teacher_energy'), statistics.get('training_token_count')
    if (mean.shape != (5120,) or std.shape != (5120,) or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all() or (std <= 0).any()
            or type(energy) not in (float, int) or not math.isfinite(energy) or energy <= 0
            or type(token_count) is not int or token_count < 6000):
        raise ValueError('Invalid fixed full-train raw-token statistics')
    teacher_means, teacher_rms, group_indices = {}, {}, {}
    groups = {}
    train_groups, val_groups = set(), set()
    required = {'audio_path', 'audio_sha256', 'group_id', 'target_path', 'target_sha256',
                'prefix_path', 'prefix_sha256', 'partition', 'teacher_mean',
                'teacher_centered_rms', 'teacher_length'}
    train_tokens = 0
    verified = {}
    for partition, ids, partition_groups in (('train', train, train_groups), ('val', val, val_groups)):
        for sid in ids:
            row = rows[sid]
            if (not isinstance(row, dict) or set(row) != required or row['partition'] != partition
                    or type(row['group_id']) is not str or not row['group_id']
                    or type(row['teacher_length']) is not int or row['teacher_length'] < 1
                    or type(row['teacher_centered_rms']) not in (int, float)
                    or not math.isfinite(row['teacher_centered_rms']) or row['teacher_centered_rms'] < 0):
                raise ValueError('Invalid direct teacher table row: ' + sid)
            vector = torch.tensor(row['teacher_mean'], dtype=torch.float32)
            if vector.shape != (5120,) or not torch.isfinite(vector).all():
                raise ValueError('Require the full finite raw-y teacher mean: ' + sid)
            teacher_means[sid], teacher_rms[sid] = vector, float(row['teacher_centered_rms'])
            group = row['group_id']
            groups.setdefault(group, len(groups))
            group_indices[sid] = groups[group]
            partition_groups.add(group)
            if partition == 'train':
                train_tokens += row['teacher_length']
            for kind in ('audio', 'target', 'prefix'):
                source, checksum = Path(row[kind + '_path']), row[kind + '_sha256']
                if (type(checksum) is not str or len(checksum) != 64
                        or any(c not in '0123456789abcdef' for c in checksum)):
                    raise ValueError('Invalid manifest source SHA: ' + sid + ':' + kind)
                if kind == 'audio' and source.suffix != '.wav':
                    raise ValueError('Deployment load_audio requires its actual .wav source')
                key = str(source.resolve())
                if key in verified and verified[key] != checksum:
                    raise ValueError('Conflicting source SHA in manifest: ' + key)
                verified[key] = checksum
    if train_groups & val_groups or train_tokens != token_count:
        raise ValueError('Train/validation song groups overlap or training token count differs')
    if verify_sources:
        file_hashes = parallel_file_sha256(verified)
        for path, checksum in verified.items():
            if checked_file_sha256(path, file_hashes) != checksum:
                raise ValueError('Changed verified direct source: ' + path)
    # All source bytes are SHA-verified; neither target raw-y nor cached P is torch-loaded.
    return dict(metadata=metadata, rows=rows, train=train, val=val, mean=mean, std=std,
                teacher_means=teacher_means, teacher_rms=teacher_rms, groups=group_indices,
                teacher_energy=float(energy), source_files=verified)


def _direct_wave_batch(ids, bundle, device):
    from music_clip_bridge import load_audio
    waves = []
    for sid in ids:
        path = Path(bundle['rows'][sid]['audio_path'])
        value = load_audio(path.stem, path.parent)[0]
        if value.shape != (2, 162000) or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError('Actual deployment waveform loader returned invalid audio: ' + sid)
        waves.append(value)
    return torch.stack(waves).to(device=device)


def _direct_targets(ids, bundle, device):
    if not ids:
        return torch.empty(0, 5120, device=device), torch.empty(0, device=device), []
    return (torch.stack([bundle['teacher_means'][sid] for sid in ids]).to(device),
            torch.tensor([bundle['teacher_rms'][sid] for sid in ids], device=device),
            [bundle['groups'][sid] for sid in ids])


def _direct_rng_state(device):
    state = dict(python=random.getstate(), torch=torch.get_rng_state())
    if device.type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state(device)
    return state


def _direct_restore_rng(state, device):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'], device)


def _direct_atomic(path, content, owned, checkpoint_model=None):
    import os
    import uuid
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    if path.exists() and path.name not in owned:
        raise FileExistsError('Refuse to replace an output not owned by this run: ' + str(path))
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    if checkpoint_model is not None:
        from music_text_adapter import save_direct_music_text_checkpoint
        save_direct_music_text_checkpoint(temporary, checkpoint_model)
    elif path.suffix == '.pt':
        with temporary.open('xb') as stream:
            torch.save(content, stream)
    else:
        with temporary.open('x') as stream:
            json.dump(content, stream, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)
    owned.add(path.name)


@torch.no_grad()
def _direct_evaluate(model, ids, bundle, device, local_batch, rank, world, moment_weight):
    model.eval()
    names = _direct_rank_ids(ids, rank, world)
    means, rms_values = [], []
    for start in range(0, len(names), local_batch):
        _refresh_online_reservation()
        wave = _direct_wave_batch(names[start:start + local_batch], bundle, device)
        raw, mask = model(wave)
        vector, rms = _direct_sample_statistics(raw, mask)
        means.append(vector)
        rms_values.append(rms)
    student_mean = torch.cat(means) if means else torch.empty(0, 5120, device=device)
    student_rms = torch.cat(rms_values) if rms_values else torch.empty(0, device=device)
    teacher_mean, teacher_rms, groups = _direct_targets(names, bundle, device)
    student_mean, student_rms = _direct_gather(student_mean), _direct_gather(student_rms)
    teacher_mean, teacher_rms = _direct_gather(teacher_mean), _direct_gather(teacher_rms)
    groups = _direct_gather(torch.tensor(groups, device=device, dtype=torch.long)).tolist()
    loss, terms, student, teacher = _direct_losses(
        student_mean, student_rms, teacher_mean, teacher_rms,
        bundle['mean'].to(device), bundle['std'].to(device), groups, moment_weight)
    logits, _, _ = _direct_clip_scores(student_mean, teacher_mean, bundle['mean'].to(device),
                                      bundle['std'].to(device), groups)
    labels = torch.arange(len(ids), device=device)
    metrics = dict(candidates=len(ids), loss=float(loss), **{k: float(v) for k, v in terms.items()})
    for k in (1, 5, 10):
        recalls = [(matrix.topk(min(k, len(ids)), dim=1).indices == labels[:, None]).any(1).float().mean()
                   for matrix in (logits, logits.T)]
        metrics['R@' + str(k)] = float(sum(recalls) / 2)
    metrics.update(student_embedding_variance=float(student.var(0, unbiased=False).mean()),
                   student_mean_variance=float(student_mean.var(0, unbiased=False).mean()),
                   student_centered_rms_mean=float(student_rms.mean()),
                   teacher_centered_rms_mean=float(teacher_rms.mean()),
                   raw_scale_ratio=float((student_rms + 1e-6).mean() / (teacher_rms + 1e-6).mean()),
                   paired_cosine=float((student * teacher).sum(-1).mean()))
    metrics['finite_noncollapsed'] = bool(
        all(math.isfinite(v) for v in metrics.values())
        and metrics['student_embedding_variance'] > 1e-10
        and metrics['student_mean_variance'] > 1e-10
        and metrics['student_centered_rms_mean'] > 1e-6)
    return metrics


def _direct_plateau_update(scheduler, validation, epoch, optimizer, minimum, patience, delta, formal=True):
    score = validation['clip']
    improved = scheduler['best_clip'] is None or score < scheduler['best_clip'] * (1 - delta)
    if improved:
        scheduler.update(best_clip=score, best_epoch=epoch, best_r1=validation['R@1'], bad_epochs=0)
    else:
        scheduler['bad_epochs'] += 1
    converged = False
    event = None
    if (formal and epoch >= minimum and scheduler['bad_epochs'] >= patience
            and validation['finite_noncollapsed']):
        if scheduler['lr_reductions'] == 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.1
            scheduler.update(lr_reductions=1, bad_epochs=0,
                             phase_anchor_r1=validation['R@1'], phase_anchor_epoch=epoch)
            event = 'learning_rate_reduced_x0.1'
        elif validation['R@1'] >= scheduler['phase_anchor_r1'] - 0.01:
            converged = True
            event = 'converged_representation_not_video_quality'
        else:
            event = 'convergence_unconfirmed_retrieval_degraded'
    return improved, converged, event


def _direct_training(args):
    import os
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from music_text_adapter import load_direct_music_text_checkpoint
    if not torch.cuda.is_available():
        raise RuntimeError('Direct training requires actual GPU; use --direct_clip_cpu_test for fixtures')
    world, rank, local_rank = (int(os.environ.get(key, default))
                               for key, default in (('WORLD_SIZE', '1'), ('RANK', '0'), ('LOCAL_RANK', '0')))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        from datetime import timedelta
        dist.init_process_group('nccl', timeout=timedelta(hours=4))
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    bundle = _direct_load_manifest(args.data_manifest, verify_sources=rank == 0)
    if world > 1:
        dist.barrier()
    train = bundle['train'][:args.limit_train] if args.limit_train else bundle['train']
    probe = train[:min(args.train_probe_count, len(train))]
    formal = not args.limit_train and not args.smoke_steps
    if formal and (world != 4 or args.local_batch_size * world < 2):
        raise ValueError('Formal protocol requires torchrun 4 ranks and at least two actual global paired examples')
    if rank == 0 and not args.out.exists():
        if not args.out.parent.is_dir():
            raise FileNotFoundError(args.out.parent)
        args.out.mkdir()
    if world > 1:
        dist.barrier()
    configuration = dict(moment_weight=args.moment_weight, local_batch_size=args.local_batch_size,
                         backbone_lr=args.backbone_lr, adapter_lr=args.adapter_lr, seed=args.seed,
                         min_epochs=args.min_epochs, patience=args.patience, rel_delta=args.rel_delta,
                         limit_train=args.limit_train, train_probe_ids=probe, world_size=world)
    source_paths = [Path(__file__), Path(__file__).with_name('music_text_adapter.py'),
                    Path(__file__).with_name('music_encoder_adapter.py'),
                    Path(__file__).with_name('music_clip_bridge.py'),
                    Path(__file__).with_name('music_text_align.py')]
    fingerprints = dict(sources={str(path.resolve()): sha256(path) for path in source_paths},
                        manifest_sha256=sha256(args.data_manifest))
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location='cpu', weights_only=True)
        if (resume.get('protocol') != DIRECT_CLIP_PROTOCOL or resume.get('configuration') != configuration
                or resume.get('source_fingerprints', {}).get('sources') != fingerprints['sources']
                or resume.get('source_fingerprints', {}).get('manifest_sha256') != fingerprints['manifest_sha256']
                or Path(resume.get('out', '')).resolve() != args.out.resolve()
                or resume.get('world_size') != world or len(resume.get('rank_rng_states', [])) != world):
            raise ValueError('Resume run/configuration/source/sampler fingerprints mismatch')
        checkpoint = args.out / 'last.pt'
        if sha256(checkpoint) != resume['checkpoint_sha256']:
            raise ValueError('Resume last model checkpoint/state SHA mismatch')
        # Load the actual saved updated complete model, never the original init.
        model = load_direct_music_text_checkpoint(checkpoint).to(device)
        fingerprints = resume['source_fingerprints']
        owned = set(resume['owned_files'])
        history, scheduler = resume['history'], resume['scheduler']
        epoch, cursor, steps = resume['epoch'], resume['epoch_cursor'], resume['optimizer_steps']
    else:
        if args.init is None:
            raise ValueError('Fresh direct training requires --init complete format4 checkpoint')
        if any(args.out.iterdir()):
            raise FileExistsError('Fresh output must be empty and dedicated to this run')
        model = load_direct_music_text_checkpoint(args.init).to(device)
        fingerprints['init_sha256'] = sha256(args.init)
        owned, history = set(), []
        scheduler = dict(best_clip=None, best_epoch=None, best_r1=None, bad_epochs=0,
                         lr_reductions=0, phase_anchor_r1=None, phase_anchor_epoch=None)
        epoch, cursor, steps = 1, 0, 0
    if model.freeze_backbone:
        raise ValueError('Both moment-weight configurations must train the dense backbone')
    model.configure_online_training(True)
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in model.backbone.parameters() if p.requires_grad],
             lr=args.backbone_lr, name='backbone'),
        dict(params=list(model.adapter.parameters()), lr=args.adapter_lr, name='adapter')], weight_decay=0.01)
    if resume:
        optimizer.load_state_dict(resume['optimizer'])
        _direct_restore_rng(resume['rank_rng_states'][rank], device)
        expected_batches = _direct_epoch_batches(train, args.local_batch_size, world, args.seed, epoch)
        sampler = resume['sampler_states'][rank]
        if sampler != dict(epoch=epoch, cursor=cursor, seed=args.seed,
                           batches=expected_batches, rank=rank, world_size=world):
            raise ValueError('Resume actual epoch order/cursor differs')
    forward = DDP(model, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else model
    plan = dict(protocol=DIRECT_CLIP_PROTOCOL, configuration=configuration, source_fingerprints=fingerprints,
                epochs_cap=args.epochs, formal=formal, train_ids=train, val_ids=bundle['val'],
                train_probe_scope='fixed_subset_not_full_training_evaluation',
                objective='bidirectional_global_BxB_CLIP_temp0.07 + weight*(paired_mean_MSE/train_std + paired_log_scalar_centered_RMS_ratio_squared)',
                forward_branches=1, teacher_table_only=True, cached_prefix_loaded=False,
                feature_cache_used=False, output='unmodified_raw_2T_5120', dtype='float32',
                global_loss_ddp_scaling='allgather_backward_SUM_cancels_DDP_gradient_MEAN',
                tail_batch_policy='balanced_actual_ID_rows_no_drop_no_repetition_communication_padding_excluded',
                global_batch_size=args.local_batch_size * world,
                source_verification='full_bytes_SHA256_parallel_16_workers_with_file_version_checks',
                distributed_timeout_seconds=14400,
                convergence_retrieval_guard='R@1 >= first_plateau_R@1 - 0.01_absolute',
                full_epoch_last_batch_sizes=[len(_direct_rank_ids(
                    _direct_epoch_batches(train, args.local_batch_size, world, args.seed, 1)[-1], r, world))
                    for r in range(world)])
    if rank == 0 and not resume:
        _direct_atomic(args.out / 'plan.json', plan, owned)
    converged = bool(resume and resume.get('converged'))
    reason = 'converged_representation_not_video_quality' if converged else 'convergence_unconfirmed'
    call_steps = 0
    initial_buffers = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
    while epoch <= args.epochs and not converged:
        batches = _direct_epoch_batches(train, args.local_batch_size, world, args.seed, epoch)
        epoch_terms, epoch_items = [], 0
        while cursor < len(batches):
            global_ids = batches[cursor]
            names = _direct_rank_ids(global_ids, rank, world)
            # Empty rank: participate in the same DDP forward/backward, but its
            # dummy source is explicitly excluded before global loss collection.
            forward_names = names if names else global_ids[:1]
            actual = torch.tensor([True] * len(names) if names else [False], device=device)
            wave = _direct_wave_batch(forward_names, bundle, device)
            teacher_mean, teacher_rms, groups = _direct_targets(names, bundle, device)
            forward.train()
            optimizer.zero_grad(set_to_none=True)
            raw, mask = forward(wave)
            loss, terms, _, _ = _direct_global_loss(
                raw, mask, teacher_mean, teacher_rms, groups,
                bundle['mean'].to(device), bundle['std'].to(device), args.moment_weight, actual)
            loss.backward()
            if steps == 0:
                gradient = _gradient_scope(model)
                if rank == 0:
                    _direct_atomic(args.out / 'gradient_step1.json', gradient, owned)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           1., error_if_nonfinite=True)
            optimizer.step()
            cursor += 1
            steps += 1
            call_steps += 1
            epoch_items += len(global_ids)
            record = dict(loss=float(loss.detach()), **{k: float(v.detach()) for k, v in terms.items()})
            epoch_terms.append((len(global_ids), record))
            if rank == 0 and (cursor == 1 or cursor % 10 == 0):
                print(json.dumps(dict(event='direct_clip_train', epoch=epoch, epoch_cursor=cursor,
                                      optimizer_steps=steps, global_actual_batch=len(global_ids), **record)), flush=True)
            del raw, mask, loss, terms, wave
            if args.smoke_steps and call_steps >= args.smoke_steps:
                break
        completed_epoch = cursor == len(batches)
        validation = _direct_evaluate(model, bundle['val'], bundle, device,
                                      args.local_batch_size, rank, world, args.moment_weight)
        training_probe = _direct_evaluate(model, probe, bundle, device,
                                          args.local_batch_size, rank, world, args.moment_weight)
        improved, converged, event = (False, False, None)
        if completed_epoch:
            improved, converged, event = _direct_plateau_update(
                scheduler, validation, epoch, optimizer, args.min_epochs,
                args.patience, args.rel_delta, formal)
        if not history:
            improved = True
        report = dict(epoch=epoch, completed_epoch=completed_epoch, epoch_cursor=cursor,
                      optimizer_steps=steps, actual_items_this_invocation_epoch=epoch_items,
                      training={key: sum(n * row[key] for n, row in epoch_terms) / max(epoch_items, 1)
                                for key in ('loss', 'clip', 'moment', 'location', 'scale')},
                      validation=validation, fixed_train_probe=training_probe,
                      scheduler=dict(scheduler), event=event,
                      learning_rates={g['name']: g['lr'] for g in optimizer.param_groups})
        history.append(report)
        if rank == 0:
            print(json.dumps(dict(event='direct_clip_evaluation', **report)), flush=True)
        for name, value in model.named_buffers():
            if not torch.equal(value.detach().cpu(), initial_buffers[name]):
                raise ValueError('Frozen backbone buffer changed: ' + name)
        control = [None]
        if rank == 0 and args.control and args.control.exists():
            control[0] = json.loads(args.control.read_text())
            if (set(control[0]) != {'protocol', 'stop_after_epoch'}
                    or control[0]['protocol'] != DIRECT_CLIP_PROTOCOL
                    or type(control[0]['stop_after_epoch']) is not int or control[0]['stop_after_epoch'] < 1):
                raise ValueError('Invalid direct rank0 controller stop command')
        if world > 1:
            dist.broadcast_object_list(control, src=0)
        stopped = bool(completed_epoch and control[0] and epoch >= control[0]['stop_after_epoch'])
        if completed_epoch:
            epoch, cursor = epoch + 1, 0
        rng = _direct_rng_state(device)
        sampler = dict(epoch=epoch, cursor=cursor, seed=args.seed,
                       batches=_direct_epoch_batches(train, args.local_batch_size, world, args.seed, epoch),
                       rank=rank, world_size=world)
        rngs, samplers = [rng], [sampler]
        if world > 1:
            rngs, samplers = [None] * world, [None] * world
            dist.all_gather_object(rngs, rng)
            dist.all_gather_object(samplers, sampler)
        if rank == 0:
            model.eval()
            _direct_atomic(args.out / 'last.pt', None, owned, checkpoint_model=model)
            if improved:
                _direct_atomic(args.out / 'best.pt', None, owned, checkpoint_model=model)
            owned.update({'train_state.pt', 'finished.json'})
            state = dict(protocol=DIRECT_CLIP_PROTOCOL, out=str(args.out.resolve()),
                         configuration=configuration, source_fingerprints=fingerprints,
                         optimizer=optimizer.state_dict(), scheduler=scheduler, epoch=epoch, epoch_cursor=cursor,
                         optimizer_steps=steps, history=history, rank_rng_states=rngs,
                         sampler_states=samplers, world_size=world, owned_files=sorted(owned),
                         checkpoint_sha256=sha256(args.out / 'last.pt'), converged=converged)
            _direct_atomic(args.out / 'train_state.pt', state, owned)
        if world > 1:
            dist.barrier()
        if converged:
            reason = 'converged_representation_not_video_quality'
            break
        if args.smoke_steps and call_steps >= args.smoke_steps:
            reason = 'smoke_convergence_unconfirmed'
            break
        if stopped:
            reason = 'controller_stop_convergence_unconfirmed'
            break
    if rank == 0:
        summary = dict(protocol=DIRECT_CLIP_PROTOCOL, complete=True, converged=converged,
                       reason=reason, optimizer_steps=steps, completed_epochs=epoch - 1,
                       checkpoint_sha256=sha256(args.out / 'last.pt'), formal=formal,
                       source_fingerprints=fingerprints, final=history[-1] if history else None,
                       scope='representation_training_not_video_quality_or_human_acceptance')
        _direct_atomic(args.out / 'finished.json', summary, owned)
        print(json.dumps(dict(event='direct_clip_finished', **summary)), flush=True)
    if world > 1:
        dist.destroy_process_group()


class _DirectCPUBackbone(torch.nn.Module):
    backend, hidden_size, sample_rate = 'musicfm', 1024, 24000

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.conv = torch.nn.Sequential(torch.nn.Linear(4, 1024), torch.nn.BatchNorm1d(1024))
        self.model.conformer = torch.nn.Module()
        self.model.conformer.layer_norm = torch.nn.LayerNorm(1024)
        self.model.conformer.layers = torch.nn.Sequential(torch.nn.Linear(1024, 1024), torch.nn.Dropout(0.5))
        self.model.stat = {'melspec_2048_mean': 0., 'melspec_2048_std': 1.}

    def forward(self, wave):
        value = F.adaptive_avg_pool1d(wave[:, None], 16).reshape(len(wave), 4, 4)
        hidden = self.model.conv(value.flatten(0, 1)).reshape(len(wave), 4, 1024)
        return self.model.conformer.layers(self.model.conformer.layer_norm(hidden))

    def metadata(self):
        return dict(backend='musicfm', model_path='cpu_explicit_fake_backbone_fixture', sample_rate=24000,
                    hidden_size=1024, parameters=sum(p.numel() for p in self.parameters()),
                    dense_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad))


def _direct_cpu_ddp_worker(rank, rendezvous, directory):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    results = []
    for count, moment_weight in ((64, 0.), (64, 1.), (7, 1.), (1, 1.)):
        torch.manual_seed(910 + count)
        reference = torch.nn.Linear(5, 5120)
        initial = {k: v.clone() for k, v in reference.state_dict().items()}
        inputs = torch.randn(count, 3, 5)
        mask = torch.ones(count, 3, dtype=torch.bool)
        mask[::2, 1] = False
        target = torch.randn(count, 5120)
        rms = torch.linspace(0.7, 1.3, count)
        groups = [i // 2 for i in range(count)]
        mean, std = torch.randn(5120) * 0.1, torch.rand(5120) + 0.5
        raw = reference(inputs)
        sm, sr = _direct_sample_statistics(raw, mask)
        reference_loss, _, _, _ = _direct_losses(sm, sr, target, rms, mean, std, groups, moment_weight)
        reference_loss.backward()
        distributed = torch.nn.Linear(5, 5120)
        distributed.load_state_dict(initial)
        wrapper = DDP(distributed)
        local = _direct_rank_ids(list(range(count)), rank, 2)
        used = local if local else [0]
        raw = wrapper(inputs[used])
        actual = torch.tensor([True] * len(local) if local else [False])
        loss, _, _, _ = _direct_global_loss(raw, mask[used], target[local], rms[local],
                                           [groups[i] for i in local], mean, std, moment_weight, actual)
        loss.backward()
        gradient_error = max(float((p.grad - q.grad).abs().max())
                             for p, q in zip(reference.parameters(), distributed.parameters()))
        torch.testing.assert_close(loss, reference_loss, rtol=1e-6, atol=1e-6)
        for p, q in zip(reference.parameters(), distributed.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=2e-6)
        results.append(dict(global_count=count, local_counts=[len(_direct_rank_ids(list(range(count)), r, 2))
                                                            for r in range(2)],
                            moment_weight=moment_weight, gradient_max_abs_error=gradient_error,
                            loss=float(loss.detach()), duplicate_groups_masked=True,
                            dummy_rank_rows_excluded=count < 2))
    if rank == 0:
        write_new(Path(directory) / 'ddp.json', results)
    dist.destroy_process_group()


def _direct_cpu_manifest_fixture(directory):
    from unittest.mock import patch
    source = Path(directory) / 'fixture.wav'
    with source.open('xb') as stream:
        stream.write(b'explicit CPU manifest SHA-only fixture; no waveform decoding')
    checksum = sha256(source)
    vector = [0.] * 5120
    train, val = ['train_' + str(i) for i in range(6000)], ['val_' + str(i) for i in range(500)]
    rows = {sid: dict(audio_path=str(source), audio_sha256=checksum,
                      target_path=str(source), target_sha256=checksum,
                      prefix_path=str(source), prefix_sha256=checksum,
                      group_id=sid, partition='train' if sid.startswith('train_') else 'val',
                      teacher_mean=vector, teacher_centered_rms=1., teacher_length=1)
            for sid in train + val}
    manifest = dict(metadata=dict(protocol=DIRECT_CLIP_PROTOCOL, complete=True, train_ids=train, val_ids=val),
                    rows=rows, statistics=dict(mean=vector, std=[1.] * 5120,
                                              teacher_energy=1., training_token_count=6000))
    with patch.object(torch, 'load', side_effect=AssertionError('raw-y/P loads are forbidden')):
        bundle = _direct_load_manifest(manifest)
    assert len(bundle['train']) == 6000 and len(bundle['val']) == 500
    del bundle
    rejection_cases = []
    for kind in ('nonfinite_teacher', 'source_sha', 'split_overlap', 'group_overlap', 'token_count'):
        saved_mean, saved_sha = rows[train[0]]['teacher_mean'], rows[train[0]]['audio_sha256']
        saved_group, saved_val = rows[val[0]]['group_id'], list(val)
        if kind == 'nonfinite_teacher':
            rows[train[0]]['teacher_mean'] = [float('nan')] + vector[1:]
        elif kind == 'source_sha':
            rows[train[0]]['audio_sha256'] = '0' * 64
        elif kind == 'split_overlap':
            val[0] = train[0]
        elif kind == 'group_overlap':
            rows[val[0]]['group_id'] = train[0]
        else:
            manifest['statistics']['training_token_count'] += 1
        try:
            _direct_load_manifest(manifest)
        except ValueError:
            rejection_cases.append(kind)
        else:
            raise AssertionError('Manifest did not reject ' + kind)
        rows[train[0]]['teacher_mean'], rows[train[0]]['audio_sha256'] = saved_mean, saved_sha
        val[:] = saved_val
        rows[val[0]]['group_id'] = saved_group
        manifest['statistics']['training_token_count'] = 6000
    return dict(locked_6000_500_finite_table_accepted=True, raw_teacher_prefix_torch_load_forbidden=True,
                actual_source_sha_verified=True, rejected=rejection_cases)


def _direct_cpu_test(report_path):
    import tempfile
    import torch.multiprocessing as mp
    from music_text_adapter import DirectMusicToH3Text, save_direct_music_text_checkpoint, load_direct_music_text_checkpoint
    torch.set_num_threads(1)
    torch.manual_seed(42)
    random.seed(42)
    model = DirectMusicToH3Text(_DirectCPUBackbone(), freeze_backbone=False)
    wave = torch.randn(4, 2, 162000)
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    model.train()
    condition = model.student_condition(wave)
    assert set(condition) == {'raw_tokens', 'valid_mask', 'tails', 'lengths'}
    assert condition['raw_tokens'].shape == (4, 8, 5120)
    assert condition['raw_tokens'].requires_grad and not any(m.training for m in model.backbone.modules())
    teacher_mean, teacher_rms = _direct_sample_statistics(condition['raw_tokens'].detach(), condition['valid_mask'])
    mean, std = torch.zeros(5120), torch.ones(5120)
    student_mean, student_rms = _direct_sample_statistics(condition['raw_tokens'], condition['valid_mask'])
    positive, _, _, _ = _direct_losses(student_mean, student_rms, teacher_mean, teacher_rms,
                                       mean, std, list(range(4)), 0.)
    wrong, _, _, _ = _direct_losses(student_mean, student_rms, teacher_mean.roll(1, 0), teacher_rms,
                                    mean, std, list(range(4)), 0.)
    assert positive < wrong
    positive.backward()
    scope = _gradient_scope(model)
    raw = torch.tensor([[[1., 3.], [100., -100.], [3., 5.]],
                        [[-2., 0.], [0., 2.], [50., 50.]]], requires_grad=True)
    mask = torch.tensor([[True, False, True], [True, True, False]])
    sm, sr = _direct_sample_statistics(raw, mask)
    torch.testing.assert_close(sm, torch.tensor([[2., 4.], [-1., 1.]]))
    torch.testing.assert_close(sr, torch.ones(2))
    location, scale = _direct_moment_terms(sm, sr, sm.detach() + 2,
                                           torch.full((2,), 2.), torch.full((2,), 2.))
    torch.testing.assert_close(location, torch.ones(2))
    expected_scale = math.log((1 + 1e-6) / (2 + 1e-6)) ** 2
    torch.testing.assert_close(scale, torch.full((2,), expected_scale))
    (location + scale).mean().backward()
    assert torch.equal(raw.grad[~mask], torch.zeros_like(raw.grad[~mask]))
    # Opposite per-sample offsets cancel in a batch mean but must incur moments.
    swapped = sm.detach().flip(0)
    l, _ = _direct_moment_terms(sm.detach(), sr.detach(), swapped, sr.detach(), torch.ones(2))
    assert l.mean() > 0 and torch.equal(sm.detach().mean(0), swapped.mean(0))
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in model.backbone.parameters() if p.requires_grad], lr=1e-5),
        dict(params=list(model.adapter.parameters()), lr=1e-4)], weight_decay=0.01)

    def step(current, opt):
        opt.zero_grad(set_to_none=True)
        augmentation = torch.randn_like(wave) * (random.random() * 0.01)
        raw_tokens, valid = current(wave + augmentation)
        a, b = _direct_sample_statistics(raw_tokens, valid)
        loss, _, _, _ = _direct_losses(a, b, teacher_mean, teacher_rms, mean, std, list(range(4)), 1.)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(current.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        return float(loss.detach())

    step(model, optimizer)
    with tempfile.TemporaryDirectory(prefix='direct_clip_cpu_') as directory:
        directory = Path(directory)
        checkpoint, state_path = directory / 'model.pt', directory / 'state.pt'
        model.eval()
        save_direct_music_text_checkpoint(checkpoint, model)
        torch.save(dict(optimizer=optimizer.state_dict(), rng=_direct_rng_state(torch.device('cpu'))), state_path)
        expected_loss = step(model, optimizer)
        expected = {key: value.clone() for key, value in model.state_dict().items()}
        fresh = load_direct_music_text_checkpoint(checkpoint, backbone=_DirectCPUBackbone())
        restored = torch.optim.AdamW([
            dict(params=[p for p in fresh.backbone.parameters() if p.requires_grad], lr=1e-5),
            dict(params=list(fresh.adapter.parameters()), lr=1e-4)], weight_decay=0.01)
        state = torch.load(state_path, map_location='cpu', weights_only=True)
        restored.load_state_dict(state['optimizer'])
        _direct_restore_rng(state['rng'], torch.device('cpu'))
        actual_loss = step(fresh, restored)
        assert expected_loss == actual_loss
        assert all(torch.equal(expected[key], fresh.state_dict()[key]) for key in expected)
        assert all(torch.equal(value, buffers[key]) for key, value in fresh.named_buffers())
        mp.spawn(_direct_cpu_ddp_worker,
                 args=('file://' + str(directory / 'gloo_store'), str(directory)), nprocs=2, join=True)
        ddp = json.loads((directory / 'ddp.json').read_text())
        manifest_fixture = _direct_cpu_manifest_fixture(directory)
    coverage = {}
    for count in (6000, 65, 63, 7, 3, 1):
        batches = _direct_epoch_batches(list(range(count)), 16, 4, 42, 1)
        flattened = [sid for batch in batches for rank in range(4) for sid in _direct_rank_ids(batch, rank, 4)]
        assert sorted(flattened) == list(range(count)) and len(flattened) == len(set(flattened))
        coverage[str(count)] = dict(actual_unique_count=len(flattened),
                                    last_rank_counts=[len(_direct_rank_ids(batches[-1], r, 4)) for r in range(4)])
    dynamic_batches = {}
    for local_batch in (1, 2, 4, 8, 16):
        batches = _direct_epoch_batches(list(range(6000)), local_batch, 4, 42, 1)
        flattened = [sid for batch in batches for r in range(4) for sid in _direct_rank_ids(batch, r, 4)]
        assert sorted(flattened) == list(range(6000)) and all(2 <= len(batch) <= local_batch * 4 for batch in batches)
        dynamic_batches[str(local_batch)] = dict(global_batch=local_batch * 4, actual_unique_count=len(flattened))
    plateau_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-4)
    scheduler = dict(best_clip=None, best_epoch=None, best_r1=None, bad_epochs=0,
                     lr_reductions=0, phase_anchor_r1=None, phase_anchor_epoch=None)
    events = []
    for epoch in range(1, 29):
        _, converged, event = _direct_plateau_update(scheduler,
            {'clip': 1., 'finite_noncollapsed': True, 'R@1': .20}, epoch, plateau_optimizer, 20, 8, .005)
        if event:
            events.append(dict(epoch=epoch, event=event))
    assert events[0]['epoch'] == 20 and events[1]['epoch'] == 28 and converged
    assert scheduler['phase_anchor_r1'] == .20 and scheduler['best_r1'] == .20
    _, degraded_converged, degraded_event = _direct_plateau_update(scheduler,
        {'clip': 1., 'finite_noncollapsed': True, 'R@1': .18}, 29, plateau_optimizer, 20, 8, .005)
    assert not degraded_converged and degraded_event == 'convergence_unconfirmed_retrieval_degraded'
    _, boundary_converged, _ = _direct_plateau_update(scheduler,
        {'clip': 1., 'finite_noncollapsed': True, 'R@1': .19}, 30, plateau_optimizer, 20, 8, .005)
    assert boundary_converged
    _, collapsed_converged, _ = _direct_plateau_update(scheduler,
        {'clip': 1., 'finite_noncollapsed': False, 'R@1': .20}, 31, plateau_optimizer, 20, 8, .005)
    assert not collapsed_converged
    summary = dict(complete=True, protocol=DIRECT_CLIP_PROTOCOL, cpu_fixture_not_actual_musicfm=True,
                   actual_gpu_training_run=False, direct_condition_four_key_graph=True,
                   positive_paired_loss=float(positive.detach()), wrong_paired_loss=float(wrong.detach()),
                   known_masked_mean_rms_exact=True, invalid_token_gradients_zero=True,
                   paired_moments_do_not_cancel_in_batch=True, known_log_rms_formula_exact=True,
                   complete_format4_roundtrip_and_adam_rng_resume_exact=True,
                   backbone_eval_bn_dropout_buffers_fixed=True, gradient_scope=scope,
                   cpu_gloo_autograd_ddp_global_loss_gradient_parity=ddp,
                   manifest_validation=manifest_fixture,
                   epoch_coverage=coverage, dynamic_formal_batch_coverage=dynamic_batches, plateau_events=events,
                   plateau_retrieval_guard=dict(first_plateau_anchor_r1=.20, absolute_allowed_drop=.01,
                                               degraded_r1=.18, degraded_convergence_rejected=True,
                                               boundary_r1=.19, boundary_convergence_accepted=True,
                                               collapsed_convergence_rejected=True),
                   source_sha256=sha256(Path(__file__)),
                   adapter_source_sha256=sha256(Path(__file__).with_name('music_text_adapter.py')))
    _direct_atomic(report_path, summary, {Path(report_path).name})
    print(json.dumps(summary), flush=True)


def direct_clip_main():
    parser = argparse.ArgumentParser(description='FP32 direct MusicFM tokens, one paired global CLIP forward')
    parser.add_argument('--direct_clip', action='store_true')
    parser.add_argument('--direct_clip_cpu_test', action='store_true')
    parser.add_argument('--data_manifest', type=Path)
    parser.add_argument('--init', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--moment_weight', '--mom_weight', type=float, choices=(0., 1.), default=0.)
    parser.add_argument('--local_batch_size', type=int, default=16)
    parser.add_argument('--backbone_lr', type=float, default=1e-5)
    parser.add_argument('--adapter_lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--min_epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--rel_delta', type=float, default=.005)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_probe_count', type=int, default=64)
    parser.add_argument('--control', type=Path)
    parser.add_argument('--smoke_steps', type=int, default=0)
    parser.add_argument('--limit_train', type=int, default=0)
    parser.add_argument('--cpu_report', type=Path, default=Path(__file__).resolve().parents[3]
                        / 'results/ms_20261020/direct_music_clip_training_cpu_v1.json')
    args = parser.parse_args()
    if args.direct_clip_cpu_test:
        return _direct_cpu_test(args.cpu_report)
    if not args.direct_clip or not args.data_manifest or not args.out:
        parser.error('Require --direct_clip --data_manifest --out and either --init or --resume')
    if (args.epochs < 1 or args.min_epochs < 20 or args.patience != 8 or args.rel_delta != .005
            or args.local_batch_size < 1 or args.train_probe_count < 2
            or args.smoke_steps < 0 or args.limit_train < 0 or args.limit_train > 6000
            or args.backbone_lr != 1e-5 or args.adapter_lr != 1e-4):
        parser.error('Require locked optimizer/plateau settings, finite epoch cap, and valid counts')
    return _direct_training(args)


def main():
    if '--direct_clip' in __import__('sys').argv or '--direct_clip_cpu_test' in __import__('sys').argv:
        return direct_clip_main()
    if '--online' in __import__('sys').argv:
        return online_main()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--targets', nargs='+', type=Path)
    parser.add_argument('--features', nargs='+', type=Path)
    parser.add_argument('--bundle', type=Path, help='SHA-verified local numerical copy of the existing2000 inputs')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--clip_weight', type=float, default=1.)
    parser.add_argument('--real_music_negatives_only', action='store_true', help='Two distinct real-music negatives; silence is evaluation only')
    parser.add_argument('--steps', type=int, default=1875)
    parser.add_argument('--eval_every', type=int, default=625)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--small_fit', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--length_weight', type=float, default=1.)
    args = parser.parse_args()
    if (args.out.exists() or not args.out.parent.is_dir() or args.steps < 1 or args.batch_size < 2
            or args.eval_every < 1 or args.small_fit < 0 or not math.isfinite(args.clip_weight) or args.clip_weight < 0
            or not math.isfinite(args.length_weight) or args.length_weight <= 0):
        parser.error('Require a new output directory and valid experiment arguments')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    random.seed(42)
    train, val = read_split(args.split)
    bundle_sha = None
    if args.bundle:
        report = json.loads(args.bundle.with_suffix('.json').read_text())
        bundle_sha = sha256(args.bundle)
        if not report['complete'] or bundle_sha != report['sha256']:
            raise ValueError('Local input bundle fails SHA validation')
        bundle = torch.load(args.bundle, map_location='cpu', weights_only=True)
        metadata = bundle['metadata']
        if (metadata['split_sha256'] != sha256(args.split) or metadata['init_sha256'] != sha256(args.init)
                or metadata['train_ids'] != train or metadata['val_ids'] != val):
            raise ValueError('Local input bundle does not match exact initialization/split')
        targets, protocols, features = bundle['targets'], bundle['protocols'], bundle['features']
        if set(targets) != set(train + val) or set(features) != set(train + val):
            raise ValueError('Local input bundle has missing or extra target/music IDs')
        mean, std, position_mean, teacher_energy = (bundle['mean'], bundle['std'], bundle['position_mean'], bundle['teacher_energy'])
    else:
        if not args.targets or not args.features:
            parser.error('Require a verified bundle or all target/feature shards')
        targets, protocols = load_targets(args.targets, set(train + val), sha256(args.split))
    model = load_music_text_checkpoint(args.init, expected_backend='musicfm').cuda()
    if not model.freeze_backbone:
        raise ValueError('This controlled experiment requires the frozen original backbone')
    if not args.bundle:
        features = load_features(args.features, set(train + val), sha256(args.split), model.backbone)
        mean, std, position_mean, teacher_energy = native_statistics(targets, train, model.adapter.qmax)
    if any(value.dtype != torch.float16 or value.shape != (2, 127, 1024) or not torch.isfinite(value).all() for value in features.values()):
        raise ValueError('Require the audited full127-frame FP16 dense feature cache')
    if (mean.shape != (5120,) or std.shape != (5120,) or position_mean.shape != (model.adapter.qmax, 5120)
            or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0)
            or not math.isfinite(teacher_energy) or teacher_energy <= 0):
        raise ValueError('Invalid train-only native H statistics')
    if args.small_fit:
        train = train[:args.small_fit]
        val = val[:min(args.small_fit, len(val))]
    mean, std, position_mean = mean.cuda(), std.cuda(), position_mean.cuda()
    with torch.no_grad():
        zero_features, zero_lengths = model.dense_features(torch.zeros(1, 2, 162000, device='cuda'))
    args.out.mkdir()
    protocol = dict(init=str(args.init), init_sha256=sha256(args.init), split_sha256=sha256(args.split),
                    train_count=len(train), val_count=len(val), steps=args.steps, batch_size=args.batch_size,
                    clip_weight=args.clip_weight, lr=args.lr, small_fit=args.small_fit, length_weight=args.length_weight,
                    normalization_from_full_train_only=True, no_extra_semantic_head=True,
                    deployed_output='unaltered_raw_native_H_tail', sampling='up_to64_evenly_spaced_valid_tokens_loss_only',
                    counterfactuals='correct/two_distinct_real_music_same_reference_context_and_indices' if args.real_music_negatives_only else 'correct/wrong_nonidentical_caption/actual_silence_same_reference_context_and_indices',
                    silence_is_eval_only=args.real_music_negatives_only,
                    loss='rawMSE/train_teacher_energy + log_length_MSE + clip_weight*same_context_sequence_InfoNCE',
                    backbone_frozen_eval=True, no_dit_video_loss=True, numerical_input_bundle_sha256=bundle_sha)
    write_new(args.out / 'plan.json', protocol)
    write_new(args.out / 'loss_statistics.pt', dict(mean=mean.cpu(), std=std.cpu(), position_mean=position_mean.cpu(),
                                                   full_train_count=len(read_split(args.split)[0]), teacher_energy=teacher_energy), binary=True)
    initial = dependence(model, val, targets, features, mean, std, position_mean, teacher_energy, zero_features, zero_lengths, args.batch_size)
    initial_fit = dependence(model, train[:min(32, len(train))], targets, features, mean, std, position_mean, teacher_energy, zero_features, zero_lengths, args.batch_size)
    initial_regression = evaluate(model.adapter, val, targets, features, args.batch_size, teacher_energy)
    write_new(args.out / 'initial_validation.json', dict(**initial, regression=initial_regression))
    write_new(args.out / 'initial_train_probe.json', initial_fit)
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=args.lr, weight_decay=0.01)
    started = time.monotonic()
    history = []
    gradient_checked = False
    items_seen = 0
    order, cursor = [], 0
    for step in range(1, args.steps + 1):
        if cursor >= len(order):
            order = random.sample(train, len(train))
            cursor = 0
        ids = order[cursor:cursor + args.batch_size]
        cursor += len(ids)
        if len(ids) < 2:
            continue
        items_seen += len(ids)
        music, context, feature_lengths, target, text_lengths, mask = batch_tensors(ids, targets, features, model.adapter.qmax)
        model.train()
        correct, logs = model.adapter(music, context, feature_lengths, query_count=target.shape[1])
        stats = masked_token_loss(correct, target, mask)
        anchor = stats['raw_mse'] / teacher_energy + args.length_weight * length_loss(logs, text_lengths, model.adapter.qmax)
        contrastive = torch.zeros((), device='cuda')
        if args.clip_weight:
            wrong_ids = mismatched_ids(ids, train, targets, offset=step % (len(train) - 1) + 1)
            wrong_music, wrong_lengths = music_tensor(wrong_ids, features)
            wrong, _ = model.adapter(wrong_music, context, wrong_lengths, query_count=target.shape[1])
            if args.real_music_negatives_only:
                second_ids = mismatched_ids(ids, train, targets, offset=step % (len(train) - 1) + 2, excluded=wrong_ids)
                second_music, second_lengths = music_tensor(second_ids, features)
                second, _ = model.adapter(second_music, context, second_lengths, query_count=target.shape[1])
                contrastive, _ = counterfactual_loss(correct, wrong, second, target, mask, mean, std)
            else:
                zero, _ = model.adapter(zero_features.expand(len(ids), -1, -1, -1), context,
                                       zero_lengths.expand(len(ids)), query_count=target.shape[1])
                contrastive, _ = counterfactual_loss(correct, wrong, zero, target, mask, mean, std)
        loss = anchor + args.clip_weight * min(1., step / 100) * contrastive
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite native-H contrastive loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not gradient_checked:
            missing = [name for name, parameter in model.adapter.named_parameters() if parameter.grad is None]
            bad = [name for name, parameter in model.adapter.named_parameters() if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
            if missing or bad or any(parameter.grad is not None for parameter in model.backbone.parameters()):
                raise ValueError(f'Bad gradient scope: {missing}, {bad}')
            gradient_checked = True
        torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(json.dumps(dict(event='train', step=step, raw_mse=float(stats['raw_mse'].detach()),
                                  infonce=float(contrastive.detach()), loss=float(loss.detach()), seconds=time.monotonic() - started)), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            val_report = dependence(model, val, targets, features, mean, std, position_mean, teacher_energy, zero_features, zero_lengths, args.batch_size)
            fit_report = dependence(model, train[:min(32, len(train))], targets, features, mean, std, position_mean, teacher_energy, zero_features, zero_lengths, args.batch_size)
            regression = evaluate(model.adapter, val, targets, features, args.batch_size, teacher_energy)
            report = dict(step=step, validation=val_report, train_probe=fit_report, regression=regression)
            history.append(report)
            write_new(args.out / f'metrics_step{step}.json', report)
            print(json.dumps(dict(event='validation', step=step, dependence=val_report['macro'], regression=regression)), flush=True)
    model.eval()
    checkpoint_path = args.out / 'last.pt'
    save_music_text_checkpoint(checkpoint_path, model)
    fresh = load_music_text_checkpoint(checkpoint_path, expected_backend='musicfm').cuda()
    music, context, feature_lengths, _, _, _ = batch_tensors(val[:1], targets, features, model.adapter.qmax)
    with torch.no_grad():
        old = model.adapter(music, context, feature_lengths)
        new = fresh.adapter(music, context, feature_lengths)
        from music_clip_bridge import load_audio
        online = fresh(load_audio(val[0])[0].cuda()[None], context)
    if not all(torch.equal(a, b) for a, b in zip(old, new)):
        raise ValueError('Complete checkpoint changes actual deployed condition')
    delta = online[0] - new[0]
    relative = float(delta.square().mean().sqrt() / new[0].square().mean().sqrt().clamp_min(1e-12))
    from music_text_adapter import predicted_lengths
    length_same = torch.equal(predicted_lengths(online[1], model.adapter.qmax), predicted_lengths(new[1], model.adapter.qmax))
    if not math.isfinite(relative) or relative > 0.01 or not length_same:
        raise ValueError('Cached/online raw H deployment gate failed')
    final = history[-1]['validation']['macro']
    gate = (final['tie_aware_correct_top1'] > 1 / 3 and final['correct_minus_wrong_score'] > 1e-3
            and final['correct_minus_zero_score'] > 1e-3 and final['correct_beats_wrong_fraction'] > 0.5
            and final['raw_mse'] <= initial['macro']['raw_mse'] * 1.05
            and history[-1]['regression']['length_mae'] <= initial_regression['length_mae'] * 1.05
            and 0.1 <= final['student_teacher_energy_ratio'] <= 2)
    summary = dict(complete=True, steps=args.steps, actual_items_seen=items_seen, protocol=protocol,
                   final=final, initial=initial['macro'], final_regression=history[-1]['regression'],
                   initial_regression=initial_regression, final_train_probe=history[-1]['train_probe']['macro'],
                   complete_checkpoint_roundtrip_exact=True, cached_online_relative_rms=relative,
                   predicted_length_same=bool(length_same), checkpoint_sha256=sha256(checkpoint_path),
                   all_adapter_layers_gradient_checked=gradient_checked, musical_dependency_gate=bool(gate),
                   cuda_peak_bytes=torch.cuda.max_memory_allocated(), seconds=time.monotonic() - started,
                   generate_only_if_musical_dependency_gate=True,
                   scope='Training/representation completed; gate is not video musical adherence or human acceptance')
    write_new(args.out / 'finished.json', summary)
    print(json.dumps(dict(event='finished', complete=True, gate=bool(gate), out=str(args.out), final=final)), flush=True)


if __name__ == '__main__':
    main()
