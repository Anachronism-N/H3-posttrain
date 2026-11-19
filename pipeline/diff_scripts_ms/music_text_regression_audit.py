"""Train-only positional mean baseline and fixed-context correct/wrong/zero music audit."""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from music_text_adapter import load_music_text_checkpoint, predicted_lengths
from music_text_align import read_split, load_targets, load_features, batch_tensors, sha256, write_new


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--targets', nargs='+', type=Path, required=True)
    parser.add_argument('--features', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    if args.out.exists() or not args.out.parent.is_dir() or args.batch_size < 1:
        parser.error('Require a new report and valid batch size')
    started = time.monotonic()
    train, val = read_split(args.split)
    targets, protocols = load_targets(args.targets, set(train + val), sha256(args.split))
    model = load_music_text_checkpoint(args.checkpoint, expected_backend='musicfm').cuda().eval()
    features = load_features(args.features, set(train + val), sha256(args.split), model.backbone)
    qmax = model.adapter.qmax
    total = torch.zeros(qmax, 5120, dtype=torch.float64)
    counts = torch.zeros(qmax, dtype=torch.float64)
    for sid in train:
        tail = targets[sid]['tail'][:qmax].double()
        total[:len(tail)] += tail
        counts[:len(tail)] += 1
    if not torch.all(counts > 0):
        raise ValueError('Training data does not support every query position')
    average = (total / counts[:, None]).float().cuda()
    global_mean = total.sum(0) / counts.sum()
    mean_length = int(round(sum(len(targets[sid]['tail']) for sid in train) / len(train)))
    zero_features, zero_lengths = model.dense_features(torch.zeros(1, 2, 162000, device='cuda'))
    summaries = []
    for offset in range(0, len(val), args.batch_size):
        ids = val[offset:offset + args.batch_size]
        music, context, feature_lengths, target, lengths, mask = batch_tensors(ids, targets, features, qmax)
        predicted, logs = model.adapter(music, context, feature_lengths, query_count=target.shape[1])
        wrong_ids = [val[(val.index(sid) + 1) % len(val)] for sid in ids]
        wrong_music = torch.stack([features[sid] for sid in wrong_ids]).cuda().float()
        wrong_counts = torch.tensor([features[sid].shape[1] for sid in wrong_ids], device='cuda')
        wrong, wrong_logs = model.adapter(wrong_music, context, wrong_counts, query_count=target.shape[1])
        zero, zero_logs = model.adapter(zero_features.expand(len(ids), -1, -1, -1), context,
                                       zero_lengths.expand(len(ids)), query_count=target.shape[1])
        for i, sid in enumerate(ids):
            n = min(int(lengths[i]), qmax)
            teacher = target[i, :n]
            expected = teacher.square().mean()
            correct_mse = (predicted[i, :n] - teacher).square().mean()
            wrong_mse = (wrong[i, :n] - teacher).square().mean()
            zero_mse = (zero[i, :n] - teacher).square().mean()
            positional_mse = (average[:n] - teacher).square().mean()
            global_mse = (global_mean.to(teacher)[None] - teacher).square().mean()
            cosine = F.cosine_similarity(predicted[i, :n], teacher, dim=-1).mean()
            summaries.append(dict(sample_id=sid, wrong_audio_id=wrong_ids[i], target_length=int(lengths[i]),
                 raw_mse=float(correct_mse), wrong_music_raw_mse=float(wrong_mse), zero_music_raw_mse=float(zero_mse),
                 train_position_mean_raw_mse=float(positional_mse), train_global_mean_raw_mse=float(global_mse),
                 mean_baseline_minus_correct_mse=float(positional_mse - correct_mse),
                 wrong_minus_correct_mse=float(wrong_mse - correct_mse),
                 zero_minus_correct_mse=float(zero_mse - correct_mse),
                 correct_target_cosine=float(cosine), teacher_energy=float(expected),
                 correct_wrong_output_rms=float((predicted[i, :n] - wrong[i, :n]).square().mean().sqrt()),
                 correct_zero_output_rms=float((predicted[i, :n] - zero[i, :n]).square().mean().sqrt()),
                 predicted_length=int(predicted_lengths(logs[i:i + 1], qmax)[0]),
                 wrong_predicted_length=int(predicted_lengths(wrong_logs[i:i + 1], qmax)[0]),
                 zero_predicted_length=int(predicted_lengths(zero_logs[i:i + 1], qmax)[0])))
        print(json.dumps(dict(event='representation_audit', count=len(summaries), expected=len(val))), flush=True)
    keys = ('raw_mse', 'wrong_music_raw_mse', 'zero_music_raw_mse', 'train_position_mean_raw_mse',
            'train_global_mean_raw_mse', 'mean_baseline_minus_correct_mse', 'wrong_minus_correct_mse',
            'zero_minus_correct_mse', 'correct_target_cosine', 'correct_wrong_output_rms', 'correct_zero_output_rms')
    macro = {key: sum(row[key] for row in summaries) / len(summaries) for key in keys}
    weights = sum(min(row['target_length'], qmax) for row in summaries)
    token_weighted = {key: sum(row[key] * min(row['target_length'], qmax) for row in summaries) / weights
                      for key in ('raw_mse', 'wrong_music_raw_mse', 'zero_music_raw_mse', 'train_position_mean_raw_mse')}
    macro['length_mae'] = sum(abs(row['predicted_length'] - row['target_length']) for row in summaries) / len(summaries)
    macro['constant_train_length_mae'] = sum(abs(mean_length - row['target_length']) for row in summaries) / len(summaries)
    result = dict(complete=len(summaries) == len(val), validation_candidates=len(val),
                  checkpoint_sha256=sha256(args.checkpoint), split_sha256=sha256(args.split),
                  train_only_position_mean_count=len(train), train_constant_length=mean_length,
                  query_max=qmax, macro=macro, token_weighted=token_weighted, cases=summaries,
                  same_reference_context_for_all_music_interventions=True, zero_uses_actual_frozen_backbone=True,
                  wrong_music='deterministic one-position cyclic shift in validation order, no self-pair',
                  seconds=time.monotonic() - started,
                  scope='Raw-token regression and music dependency only; positional word indices do not guarantee semantic timing. Not video musical accuracy or a unique choreography truth.')
    write_new(args.out, result)
    print(json.dumps(dict(event='audit_finished', complete=result['complete'], macro=macro, token_weighted=token_weighted)), flush=True)


if __name__ == '__main__':
    main()
