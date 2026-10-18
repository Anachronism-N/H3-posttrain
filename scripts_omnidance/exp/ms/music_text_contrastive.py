"""Raw native H sequence alignment with same-image music counterfactual InfoNCE."""
import argparse
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from music_text_adapter import load_music_text_checkpoint, save_music_text_checkpoint, length_loss, masked_token_loss
from music_text_align import read_split, load_targets, load_features, batch_tensors, evaluate, sha256, write_new


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


def _online_batch(ids, bundle, qmax):
    wave = torch.stack([bundle['waveforms'][sid] for sid in ids]).cuda()
    context = torch.stack([bundle['targets'][sid]['prefix_context'] for sid in ids]).cuda().float()
    tails = [bundle['targets'][sid]['tail'] for sid in ids]
    if any(len(value) > qmax for value in tails):
        raise ValueError('Teacher exceeds the train-only query bound')
    target = torch.stack([F.pad(value, (0, 0, 0, qmax - len(value))) for value in tails]).cuda().float()
    lengths = torch.tensor([len(value) for value in tails], device='cuda')
    teacher_mask = torch.arange(qmax, device='cuda')[None] < lengths[:, None]
    return wave, context, target, lengths, teacher_mask


def _online_scores(conditions, teacher, teacher_mask, mean, std):
    return torch.stack([score_samples(c['raw_tokens'], teacher, teacher_mask, mean, std,
                                     student_mask=c['valid_mask']) for c in conditions], dim=1)


def _online_conditions(model, ids, candidates, bundle, context, offset):
    targets = bundle['targets']
    first = mismatched_ids(ids, candidates, targets, offset=offset)
    second = mismatched_ids(ids, candidates, targets, offset=offset + 1, excluded=first)
    outputs = []
    for names in (ids, first, second):
        wave = torch.stack([bundle['waveforms'][sid] for sid in names]).cuda()
        outputs.append(model.student_condition(wave, context))
    return outputs, first, second


@torch.no_grad()
def _evaluate_online(model, ids, candidates, bundle, mean, std, position_mean, batch_size):
    model.eval()
    rows = []
    raw_sum, wrong_sum, zero_sum, teacher_sum, student_sum, count = 0., 0., 0., 0., 0., 0
    for start in range(0, len(ids), batch_size):
        _refresh_online_reservation()
        names = ids[start:start + batch_size]
        wave, context, target, lengths, mask = _online_batch(names, bundle, model.adapter.qmax)
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
                length_absolute_error=abs(int(conditions[0]['lengths'][i]) - int(lengths[i]))))
    macro = {key: sum(row[key] for row in rows) / len(rows) for key in (
        'correct_minus_wrong_score', 'correct_minus_second_score', 'correct_minus_zero_score',
        'tie_aware_top1', 'raw_mse', 'wrong_raw_mse', 'zero_raw_mse', 'position_mean_mse',
        'correct_wrong_output_rms', 'length_absolute_error')}
    macro.update(correct_beats_wrong_fraction=sum(r['correct_minus_wrong_score'] > 1e-6 for r in rows) / len(rows),
                 correct_beats_second_fraction=sum(r['correct_minus_second_score'] > 1e-6 for r in rows) / len(rows),
                 raw_mse_tokenweighted=raw_sum / count, wrong_mse_tokenweighted=wrong_sum / count,
                 zero_mse_tokenweighted=zero_sum / count, student_teacher_energy_ratio=student_sum / max(teacher_sum, 1e-12))
    return dict(candidates=len(rows), macro=macro, cases=rows, student_uses_predicted_valid_interval=True,
                teacher_only_controls_supervision=True, silence_is_eval_only=True,
                scope='Representation diagnostics, not dance or video accuracy')


def _online_deployment_audit(model, checkpoint, wave, context, out):
    import os
    import subprocess
    import tempfile
    from music_text_adapter import load_music_text_checkpoint
    model.train()
    with torch.enable_grad():
        train_condition = model.student_condition(wave[:1], context[:1])
    if not train_condition['raw_tokens'].requires_grad or not train_condition['log_length'].requires_grad:
        raise ValueError('Training student condition lost its graph')
    expected = {key: train_condition[key].detach().cpu() for key in ('raw_tokens', 'log_length', 'lengths', 'valid_mask')}
    expected_tail = train_condition['tails'][0].detach().cpu()
    del train_condition
    model.eval()
    with torch.no_grad():
        actual = model.student_condition(wave[:1], context[:1])
        if not all(torch.equal(expected[key], actual[key].cpu()) for key in expected):
            raise ValueError('Train/eval student construction differs')
    del actual
    fresh = load_music_text_checkpoint(checkpoint, expected_backend='musicfm').cuda().eval()
    with torch.no_grad():
        actual = fresh.student_condition(wave[:1], context[:1])
        if not all(torch.equal(expected[key], actual[key].cpu()) for key in expected):
            raise ValueError('Full checkpoint changes online student condition')
    del actual, fresh
    model.cpu()
    torch.cuda.empty_cache()
    with tempfile.TemporaryDirectory(prefix='music_online_deploy_') as directory:
        inp, result = Path(directory) / 'input.pt', Path(directory) / 'result.pt'
        torch.save(dict(waveform=wave[0].cpu(), context=context[0].cpu()), inp)
        command = [__import__('sys').executable, str(Path(__file__).with_name('music_text_infer.py')),
                   '--checkpoint', str(checkpoint), '--checkpoint_sha256', sha256(checkpoint),
                   '--input', str(inp), '--out', str(result)]
        completed = subprocess.run(command, capture_output=True, text=True, env=os.environ.copy())
        write_new(out / 'independent_deployment_process.json', dict(command=command, returncode=completed.returncode,
                    stdout=completed.stdout, stderr=completed.stderr))
        if completed.returncode:
            raise RuntimeError('Independent deployment failed: ' + completed.stderr[-4000:])
        deployed = torch.load(result, map_location='cpu', weights_only=True)
    if not torch.equal(expected_tail, deployed['tail']) or int(expected['lengths'][0]) != deployed['predicted_length']:
        raise ValueError('Independent inference changes actual deployed tail')
    # H3 prefix/tags/packed parity is audited separately in the H3 environment.
    write_new(out / 'deployment_probe.pt', dict(waveform=wave[0].cpu(), context=context[0].cpu(),
              tail=expected_tail, length=int(expected['lengths'][0]), checkpoint_sha256=sha256(checkpoint)), binary=True)
    return dict(train_eval_raw_length_mask_exact=True, complete_checkpoint_condition_exact=True,
                independent_inference_tail_exact=True, training_graph_present=True,
                h3_prefix_tags_packed_checked=False)


def online_main():
    parser = argparse.ArgumentParser(description='Online MusicFM dense backbone + complete raw H adapter')
    parser.add_argument('--online', action='store_true', required=True)
    parser.add_argument('--online_bundle', type=Path, required=True)
    parser.add_argument('--init', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--backbone_train', action='store_true')
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
    if set(bundle) != {'metadata', 'targets', 'waveforms', 'mean', 'std', 'position_mean', 'teacher_energy', 'protocols'}:
        raise ValueError('Online bundle must contain teacher labels and waveforms, never cached student features')
    train, val = read_split(args.split)
    metadata = bundle['metadata']
    if (metadata['format_version'] != 1 or metadata['split_sha256'] != sha256(args.split)
            or metadata['train_ids'] != train or metadata['val_ids'] != val
            or metadata['init_sha256'] != sha256(args.init)
            or metadata['student_feature_cache_used'] is not False
            or set(bundle['targets']) != set(train + val) or set(bundle['waveforms']) != set(train + val)):
        raise ValueError('Online bundle/initialization/split mismatch')
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
    model.configure_online_training(args.backbone_train)
    qmax = model.adapter.qmax
    if position_mean.shape != (qmax, 5120) or max(len(bundle['targets'][sid]['tail']) for sid in train) != qmax:
        raise ValueError('Query bound must come from full train only')
    if args.small_fit:
        train = train[:args.small_fit]
        val = val[:min(len(val), args.small_fit)]
    if args.validation_count:
        val = val[:args.validation_count]
    args.out.mkdir()
    protocol = dict(init=str(args.init), init_sha256=sha256(args.init), online_bundle_sha256=bundle_hash,
        split_sha256=sha256(args.split), train_ids=train, validation_ids=val, steps=args.steps,
        batch_size=args.batch_size, microbatch=args.microbatch, accumulation=args.batch_size // args.microbatch,
        backbone_train=args.backbone_train, backbone_lr=args.backbone_lr, adapter_lr=args.adapter_lr,
        clip_weight=args.clip_weight, length_weight=args.length_weight, seed=42,
        shared_student_condition=True, student_uses_ground_truth_length=False,
        online_waveform_backbone_all_three_branches=True, student_feature_cache_used=False,
        negatives='two_different_ids_and_caption_hashes_not_verified_distinct_source_songs',
        backbone_eval_behavior_with_parameter_gradients=args.backbone_train,
        dtype='float32_no_cached_feature_roundtrip', no_dit_video_loss=True,
        source_sha256=sha256(Path(__file__)), adapter_source_sha256=sha256(Path(__file__).with_name('music_text_adapter.py')),
        loss='rawMSE/train_teacher_energy + length_weight*log_length_MSE + clip_weight*min(step/100,1)*same_context_sequence_NCE',
        sampling='independent_up_to64_student_predicted_valid_tokens_and_teacher_valid_tokens',
        full_train_statistics_reused=True, small_fit=args.small_fit)
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
        totals = dict(loss=0., raw_mse=0., length=0., infonce=0.)
        batch_tokens = sum(len(bundle['targets'][sid]['tail']) for sid in ids)
        for start in range(0, len(ids), args.microbatch):
            names = ids[start:start + args.microbatch]
            wave, context, target, lengths, mask = _online_batch(names, bundle, qmax)
            conditions, _, _ = _online_conditions(model, names, train, bundle, context, step % (len(train) - 1) + 1)
            mse = masked_token_loss(conditions[0]['raw_tokens'], target, mask)['raw_mse']
            ll = length_loss(conditions[0]['log_length'], lengths, qmax)
            scores = _online_scores(conditions, target, mask, mean, std)
            nce = F.cross_entropy(scores / 0.07, torch.zeros(len(names), dtype=torch.long, device='cuda'))
            weight = len(names) / len(ids)
            token_weight = int(mask.sum()) / batch_tokens
            loss = mse / teacher_energy * token_weight + weight * (
                args.length_weight * ll + args.clip_weight * min(1., step / 100) * nce)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite online training loss')
            loss.backward()
            totals['loss'] += float(loss.detach())
            totals['raw_mse'] += float(mse.detach()) * token_weight
            totals['length'] += float(ll.detach()) * weight
            totals['infonce'] += float(nce.detach()) * weight
            del conditions, loss, mse, ll, scores, nce
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
    wave, context, _, _, _ = _online_batch(val[:1], bundle, qmax)
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


def main():
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
