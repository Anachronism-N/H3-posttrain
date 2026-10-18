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


def score_samples(student, teacher, valid_mask, mean, std, limit=64, temperature=0.1):
    if (student.ndim != 3 or student.shape != teacher.shape or student.shape[-1] != 5120
            or valid_mask.shape != student.shape[:2] or valid_mask.dtype != torch.bool):
        raise ValueError('Expected matching actual raw H sequences and valid token mask')
    if limit < 1 or temperature <= 0 or not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError('Invalid sequence score inputs')
    scores = []
    for a, b, mask in zip(student, teacher, valid_mask):
        indices = mask.nonzero().flatten()
        if not len(indices):
            raise ValueError('No valid teacher tokens')
        if len(indices) > limit:
            selected = torch.linspace(0, len(indices) - 1, limit, device=indices.device).round().long()
            indices = indices[selected]
        # Fixed train-only normalization is for the loss, never the deployed raw condition.
        x = F.normalize((a.index_select(0, indices).float() - mean) / std, dim=-1)
        y = F.normalize((b.index_select(0, indices).detach().float() - mean) / std, dim=-1)
        similarity = x @ y.T
        row = temperature * (torch.logsumexp(similarity / temperature, dim=-1) - math.log(len(indices)))
        col = temperature * (torch.logsumexp(similarity / temperature, dim=-2) - math.log(len(indices)))
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


def mismatched_ids(ids, all_ids, targets, offset=1):
    positions = {sid: i for i, sid in enumerate(all_ids)}
    result = []
    for sid in ids:
        for shift in range(offset, offset + len(all_ids)):
            candidate = all_ids[(positions[sid] + shift) % len(all_ids)]
            if candidate != sid and targets[candidate]['metadata']['caption_sha256'] != targets[sid]['metadata']['caption_sha256']:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--targets', nargs='+', type=Path)
    parser.add_argument('--features', nargs='+', type=Path)
    parser.add_argument('--bundle', type=Path, help='SHA-verified local numerical copy of the existing2000 inputs')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--clip_weight', type=float, default=1.)
    parser.add_argument('--steps', type=int, default=1875)
    parser.add_argument('--eval_every', type=int, default=625)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--small_fit', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0003)
    args = parser.parse_args()
    if (args.out.exists() or not args.out.parent.is_dir() or args.steps < 1 or args.batch_size < 2
            or args.eval_every < 1 or args.small_fit < 0 or not math.isfinite(args.clip_weight) or args.clip_weight < 0):
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
                    clip_weight=args.clip_weight, lr=args.lr, small_fit=args.small_fit,
                    normalization_from_full_train_only=True, no_extra_semantic_head=True,
                    deployed_output='unaltered_raw_native_H_tail', sampling='up_to64_evenly_spaced_valid_tokens_loss_only',
                    counterfactuals='correct/wrong_nonidentical_caption/actual_silence_same_reference_context_and_indices',
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
        anchor = stats['raw_mse'] / teacher_energy + length_loss(logs, text_lengths, model.adapter.qmax)
        contrastive = torch.zeros((), device='cuda')
        if args.clip_weight:
            wrong_ids = mismatched_ids(ids, train, targets, offset=step % (len(train) - 1) + 1)
            wrong_music, wrong_lengths = music_tensor(wrong_ids, features)
            wrong, _ = model.adapter(wrong_music, context, wrong_lengths, query_count=target.shape[1])
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
