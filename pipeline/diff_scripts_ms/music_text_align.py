"""Frozen MusicFM + raw-token adapter; no H3/video loss or caption retrieval at deployment."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from music_encoder_adapter import DenseMusicEncoder
from music_clip_bridge import load_audio
from music_text_adapter import (MusicTextAdapter, MusicToH3Text, masked_token_loss,
                               length_loss, predicted_lengths, save_music_text_checkpoint,
                               load_music_text_checkpoint, PREPROCESSING)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value, binary=False):
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    with path.open('xb' if binary else 'x') as stream:
        if binary:
            torch.save(value, stream)
        else:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')


def read_split(path):
    split = json.loads(Path(path).read_text())
    train, val = split['train'], split['val']
    if not train or not val or len(set(train)) != len(train) or len(set(val)) != len(val) or set(train) & set(val):
        raise ValueError('Require unique, disjoint, nonempty train/val lists')
    if any(Path(sid).name != sid or sid in ('.', '..') for sid in train + val):
        raise ValueError('Invalid sample ID')
    return train, val


def original_backbone(path):
    return DenseMusicEncoder('musicfm', str(path)).cuda().eval().requires_grad_(False)


def feature_cache(args):
    train, val = read_split(args.split)
    ids = (train + val)[args.shard::args.nshard]
    if args.max_items:
        ids = ids[:args.max_items]
    if args.out.exists() or not args.out.parent.is_dir():
        raise FileExistsError('Require a new feature-cache output path')
    backbone = original_backbone(args.model_path)
    wrapper = MusicToH3Text(backbone, MusicTextAdapter(1).cuda(), freeze_backbone=True).eval()
    features, lengths, wave_hashes = {}, {}, {}
    started = time.monotonic()
    with torch.no_grad():
        for offset in range(0, len(ids), args.batch_size):
            batch_ids = ids[offset:offset + args.batch_size]
            waveforms = torch.stack([load_audio(sid)[0] for sid in batch_ids]).cuda()
            # The shared 32k loader always returns the same 162000-sample padded waveform.
            # Both feature caching and future deployment encode this exact complete input.
            values, valid = wrapper.dense_features(waveforms)
            for sid, value, count, waveform in zip(batch_ids, values, valid, waveforms):
                features[sid] = value[:, :int(count)].cpu().half()
                lengths[sid] = int(count)
                wave_hashes[sid] = hashlib.sha256(waveform.cpu().numpy().tobytes()).hexdigest()
            print(json.dumps(dict(event='features', count=min(offset + args.batch_size, len(ids)),
                                  expected=len(ids), elapsed=time.monotonic() - started)), flush=True)
    metadata = dict(complete=True, count=len(ids), split_sha256=sha256(args.split),
                    ids=ids, preprocessing=PREPROCESSING, backbone=backbone.metadata(),
                    backbone_weights_sha256=sha256(args.model_path / 'pretrained_msd.pt'),
                    backbone_stats_sha256=sha256(args.model_path / 'msd_stats.json'),
                    waveform_loader='music_clip_bridge.load_audio', padded_32k_samples=162000,
                    shard=args.shard, nshard=args.nshard, explicit_subset=args.max_items,
                    lengths=lengths, waveform_sha256=wave_hashes, seconds=time.monotonic() - started)
    write_new(args.out, dict(metadata=metadata, features=features), binary=True)
    reread = torch.load(args.out, map_location='cpu', weights_only=True)
    if set(reread['features']) != set(ids) or not all(torch.equal(value, reread['features'][sid]) for sid, value in features.items()):
        raise ValueError('Feature cache roundtrip mismatch')
    print(json.dumps(dict(event='features_finished', count=len(ids), path=str(args.out), sha256=sha256(args.out))), flush=True)


def load_targets(directories, expected, split_hash):
    result = {}
    protocols = []
    shared_signature = None
    for directory in directories:
        report = json.loads((directory / 'finished.json').read_text())
        source = report['source']
        if not report['complete'] or source['split_sha256'] != split_hash:
            raise ValueError('Incomplete or mismatching target-cache shard')
        keys = ('caps_sha256', 'split_sha256', 'image_root', 'interface', 'reference_image_short_edge',
                'structural_audio_label', 'semantic_student_prompt', 'final_norm', 'caption_fields', 'teacher_transform')
        signature = {key: source.get(key) for key in keys}
        if (signature['interface'] != 'raw_h3_last_hidden_5120_tail_replacement'
                or signature['semantic_student_prompt'] != ' ' or signature['final_norm'] != 'Identity'
                or signature['teacher_transform'] not in ('none', 'gender_pronouns_to_dancer_v1')):
            raise ValueError('Unsupported teacher-token protocol')
        if shared_signature is not None and signature != shared_signature:
            raise ValueError('Teacher cache shards use different caption/transform/interface protocols')
        shared_signature = signature
        protocol_path = directory / 'plan.json'
        protocol_hash = sha256(protocol_path)
        if json.loads(protocol_path.read_text()) != source:
            raise ValueError('Teacher plan and finished report disagree')
        protocols.append(source)
        for row in report['rows']:
            sid, path = row['sample_id'], Path(row['path'])
            if sid in result:
                raise ValueError('Duplicate target ID')
            if sid not in expected:
                continue
            if sha256(path) != row['file_sha256']:
                raise ValueError('Teacher file SHA mismatch: ' + sid)
            data = torch.load(path, map_location='cpu', weights_only=True)
            if (data['metadata'].get('source_protocol_sha256') != protocol_hash
                    or Path(data['metadata'].get('source_protocol_path', '')).resolve() != protocol_path.resolve()):
                raise ValueError('Teacher record points to a different source protocol: ' + sid)
            if (data['metadata']['sample_id'] != sid or data['tail'].shape != (row['tail_length'], 5120)
                    or data['prefix_context'].shape != (5120,) or not torch.isfinite(data['tail']).all()
                    or not torch.isfinite(data['prefix_context']).all()):
                raise ValueError('Invalid teacher tokens: ' + sid)
            result[sid] = data
    if set(result) != set(expected):
        raise ValueError('Teacher shards do not cover exactly the requested training/validation IDs')
    return result, protocols


def load_features(paths, expected, split_hash, backbone):
    result = {}
    weights_sha256 = sha256(Path(backbone.model_path) / 'pretrained_msd.pt')
    stats_sha256 = sha256(Path(backbone.model_path) / 'msd_stats.json')
    for path in paths:
        state = torch.load(path, map_location='cpu', weights_only=True)
        metadata = state['metadata']
        if (not metadata['complete'] or metadata['split_sha256'] != split_hash
                or metadata['preprocessing'] != PREPROCESSING or metadata['backbone'] != backbone.metadata()
                or metadata['backbone_weights_sha256'] != weights_sha256
                or metadata['backbone_stats_sha256'] != stats_sha256):
            raise ValueError('Features do not match frozen deployment backbone/preprocessing')
        for sid, features in state['features'].items():
            if sid in result:
                raise ValueError('Duplicate feature ID')
            if sid not in expected:
                continue
            if features.ndim != 3 or features.shape[0] != 2 or features.shape[-1] != 1024 or not torch.isfinite(features).all():
                raise ValueError('Invalid music features: ' + sid)
            result[sid] = features
    if set(result) != set(expected):
        raise ValueError('Feature shards missing requested IDs')
    return result


def batch_tensors(ids, targets, features, query_max):
    n_features = [features[sid].shape[1] for sid in ids]
    n_tokens = [len(targets[sid]['tail']) for sid in ids]
    count = min(query_max, max(n_tokens))
    music = torch.stack([F.pad(features[sid], (0, 0, 0, max(n_features) - n)) for sid, n in zip(ids, n_features)]).cuda().float()
    context = torch.stack([targets[sid]['prefix_context'] for sid in ids]).cuda().float()
    target = torch.stack([F.pad(targets[sid]['tail'][:count], (0, 0, 0, max(0, count - n))) for sid, n in zip(ids, n_tokens)]).cuda().float()
    lengths = torch.tensor(n_tokens, device='cuda')
    mask = torch.arange(count, device='cuda')[None] < lengths[:, None]
    return music, context, torch.tensor(n_features, device='cuda'), target, lengths, mask


@torch.no_grad()
def evaluate(adapter, ids, targets, features, batch_size, teacher_energy):
    adapter.eval()
    accum = dict(raw_mse=0., cosine_loss=0., length_mae=0., log_length_mse=0., student_energy=0., teacher_energy=0., zero_mse=0.)
    tokens = 0
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        music, context, music_lengths, target, lengths, mask = batch_tensors(batch, targets, features, adapter.qmax)
        prediction, log_length = adapter(music, context, music_lengths, query_count=target.shape[1])
        stats = masked_token_loss(prediction, target, mask)
        n = int(mask.sum())
        tokens += n
        for key in ('raw_mse', 'cosine_loss'):
            accum[key] += float(stats[key]) * n
        accum['student_energy'] += float(prediction[mask].float().square().mean()) * n
        accum['teacher_energy'] += float(target[mask].float().square().mean()) * n
        accum['zero_mse'] += float(target[mask].square().mean()) * n
        accum['length_mae'] += float((predicted_lengths(log_length, adapter.qmax) - lengths).abs().sum())
        accum['log_length_mse'] += float((log_length.float() - lengths.float().log()).square().sum())
    result = {key: value / (len(ids) if key in ('length_mae', 'log_length_mse') else tokens) for key, value in accum.items()}
    result.update(candidates=len(ids), valid_tokens=tokens, mse_over_train_teacher_energy=result['raw_mse'] / teacher_energy,
                  student_teacher_energy_ratio=result['student_energy'] / max(result['teacher_energy'], 1e-12),
                  student_rms=math.sqrt(result['student_energy']), teacher_rms=math.sqrt(result['teacher_energy']),
                  target_tokens_above_query_max=sum(len(targets[sid]['tail']) > adapter.qmax for sid in ids),
                  scope='Raw-token representation and length, not musical generation quality')
    return result


def train_adapter(args):
    if args.out.exists() or not args.out.parent.is_dir():
        raise FileExistsError('Require a new training output directory')
    train, val = read_split(args.split)
    if args.small_fit:
        train = train[:args.small_fit]
        val = val[:min(args.small_fit, len(val))]
    expected = set(train + val)
    split_hash = sha256(args.split)
    targets, protocols = load_targets(args.targets, expected, split_hash)
    query_max = max(len(targets[sid]['tail']) for sid in train)
    backbone = original_backbone(args.model_path)
    features = load_features(args.features, expected, split_hash, backbone)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    adapter = MusicTextAdapter(query_max).cuda()
    if args.initialize_length_mean:
        mean_log_length = sum(math.log(len(targets[sid]['tail'])) for sid in train) / len(train)
        torch.nn.init.zeros_(adapter.length_head[-1].weight)
        torch.nn.init.constant_(adapter.length_head[-1].bias, mean_log_length)
    wrapper = MusicToH3Text(backbone, adapter, freeze_backbone=True)
    teacher_energy = sum(float(targets[sid]['tail'].float().square().sum()) for sid in train) / sum(targets[sid]['tail'].numel() for sid in train)
    if not math.isfinite(teacher_energy) or teacher_energy <= 0:
        raise ValueError('Invalid teacher energy')
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)
    args.out.mkdir()
    protocol = dict(split_sha256=split_hash, train_ids=train, val_ids=val, query_max=query_max,
                    teacher_energy=teacher_energy, teacher_cache_protocols=protocols, model_path=str(args.model_path),
                    model_weights_sha256=sha256(args.model_path / 'pretrained_msd.pt'),
                    cached_feature_sources=[dict(path=str(path), sha256=sha256(path)) for path in args.features],
                    source_sha256=sha256(Path(__file__)), adapter_source_sha256=sha256(Path(__file__).with_name('music_text_adapter.py')),
                    training_semantic_inputs=['music', 'reference_structural_prefix_context'],
                    student_caption_input=False, no_dit_video_loss=True, preprocessing=PREPROCESSING,
                    backbone_frozen_eval=True, steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                    small_fit=args.small_fit, seed=args.seed,
                    loss='raw_token_MSE/train_teacher_energy + cosine_weight*cosine + length_weight*log_length_MSE',
                    cosine_weight=args.cosine_weight, length_weight=args.length_weight,
                    length_bias_from_train_mean=args.initialize_length_mean,
                    shared_final_projection=True, generate_uses_predicted_length=True)
    write_new(args.out / 'plan.json', protocol)
    gradient_checked = False
    started = time.monotonic()
    best, best_step, best_forward = float('inf'), None, None
    order, cursor = [], 0
    history = []
    items_seen = 0
    initial_probe = evaluate(adapter, train[:min(32, len(train))], targets, features, args.batch_size, teacher_energy)
    write_new(args.out / 'initial_train_probe.json', initial_probe)
    for step in range(1, args.steps + 1):
        if cursor >= len(order):
            order = random.sample(train, len(train))
            cursor = 0
        ids = order[cursor:cursor + args.batch_size]
        cursor += len(ids)
        items_seen += len(ids)
        music, context, music_lengths, target, lengths, mask = batch_tensors(ids, targets, features, query_max)
        wrapper.train()
        prediction, log_length = adapter(music, context, music_lengths, query_count=target.shape[1])
        stats = masked_token_loss(prediction, target, mask)
        loss = stats['raw_mse'] / teacher_energy + args.cosine_weight * stats['cosine_loss'] + args.length_weight * length_loss(log_length, lengths, query_max)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not gradient_checked:
            missing = [name for name, parameter in adapter.named_parameters() if parameter.grad is None]
            bad = [name for name, parameter in adapter.named_parameters() if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
            if missing or bad or any(parameter.grad is not None for parameter in backbone.parameters()):
                raise ValueError(f'Invalid gradient scope missing={missing}, bad={bad}')
            gradient_checked = True
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.)
        warmup = min(1., step / min(100, max(1, args.steps // 10)))
        for group in optimizer.param_groups:
            group['lr'] = args.lr * warmup
        optimizer.step()
        if step == 1 or step % 25 == 0:
            print(json.dumps(dict(event='train', step=step, loss=float(loss.detach()),
                                  raw_mse=float(stats['raw_mse'].detach()), cosine=float(stats['cosine_loss'].detach()),
                                  elapsed=time.monotonic() - started)), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            metric = evaluate(adapter, val, targets, features, args.batch_size, teacher_energy)
            fit_metric = evaluate(adapter, train[:min(32, len(train))], targets, features, args.batch_size, teacher_energy)
            score = metric['raw_mse'] / teacher_energy + args.cosine_weight * metric['cosine_loss'] + args.length_weight * metric['log_length_mse']
            record = dict(step=step, val=metric, train_probe=fit_metric, seconds=time.monotonic() - started)
            history.append(record)
            write_new(args.out / f'metrics_step{step}.json', record)
            if score < best:
                best, best_step = score, step
                save_music_text_checkpoint(args.out / f'candidate_step{step}.pt', wrapper)
                probe_music, probe_context, probe_lengths, _, _, _ = batch_tensors(val[:1], targets, features, query_max)
                with torch.no_grad():
                    best_forward = tuple(value.detach().cpu() for value in adapter(probe_music, probe_context, probe_lengths))
            print(json.dumps(dict(event='validation', **record)), flush=True)
    if args.small_fit:
        final_probe = history[-1]['train_probe']
        length_tolerance = max(2., 0.05 * sum(len(targets[sid]['tail']) for sid in train) / len(train))
        if (final_probe['raw_mse'] >= initial_probe['raw_mse']
                or final_probe['length_mae'] > length_tolerance
                or final_probe['log_length_mse'] >= initial_probe['log_length_mse']
                or final_probe['student_teacher_energy_ratio'] < 0.1):
            raise ValueError('Small-fit token or length regression failed; refuse formal-training acceptance')
    wrapper.eval()
    final_path = args.out / 'last.pt'
    save_music_text_checkpoint(final_path, wrapper)
    # Reload the complete backbone + every adapter layer, then compare real held-out inputs.
    fresh = load_music_text_checkpoint(final_path, expected_backend='musicfm', expected_query_max=query_max).cuda()
    music, context, music_lengths, _, _, _ = batch_tensors(val[:1], targets, features, query_max)
    with torch.no_grad():
        old = adapter(music, context, music_lengths)
        new = fresh.adapter(music, context, music_lengths)
    if not all(torch.equal(a, b) for a, b in zip(old, new)):
        raise ValueError('Strict checkpoint reload changes deployment outputs')
    # Online music uses exactly the same loader and every saved backbone/adapter layer.
    waveform = load_audio(val[0])[0].cuda()[None]
    with torch.no_grad():
        online = fresh(waveform, context)
    delta = online[0] - new[0]
    relative_rms = float(delta.square().mean().sqrt() / new[0].square().mean().sqrt().clamp_min(1e-12))
    if not math.isfinite(relative_rms) or relative_rms > 0.01:
        raise ValueError('Cached/online trained adapter differs beyond declared tolerance')
    online_check = dict(relative_rms=relative_rms, max_abs=float(delta.abs().max()),
                        predicted_length_cached=int(predicted_lengths(new[1], query_max)[0]),
                        predicted_length_online=int(predicted_lengths(online[1], query_max)[0]),
                        same_loader_and_all_layers=True, exact_raw=torch.equal(online[0], new[0]))
    if online_check['predicted_length_cached'] != online_check['predicted_length_online']:
        raise ValueError('Cached/online inference predicts a different token count')
    del fresh, online
    best_path = args.out / f'candidate_step{best_step}.pt'
    selected = load_music_text_checkpoint(best_path, expected_backend='musicfm', expected_query_max=query_max).cuda()
    with torch.no_grad():
        selected_cached = selected.adapter(music, context, music_lengths)
        selected_online = selected(waveform, context)
    if not all(torch.equal(actual.cpu(), expected) for actual, expected in zip(selected_cached, best_forward)):
        raise ValueError('Selected checkpoint differs from its saved-step deployment output')
    selected_delta = selected_online[0] - selected_cached[0]
    selected_relative = float(selected_delta.square().mean().sqrt() / selected_cached[0].square().mean().sqrt().clamp_min(1e-12))
    selected_check = dict(relative_rms=selected_relative, max_abs=float(selected_delta.abs().max()),
                          predicted_length_cached=int(predicted_lengths(selected_cached[1], query_max)[0]),
                          predicted_length_online=int(predicted_lengths(selected_online[1], query_max)[0]),
                          exact_saved_step=True, checkpoint_sha256=sha256(best_path))
    if (not math.isfinite(selected_relative) or selected_relative > 0.01
            or selected_check['predicted_length_cached'] != selected_check['predicted_length_online']):
        raise ValueError('Selected checkpoint cached/online condition mismatch')
    summary = dict(complete=True, steps=args.steps, train_count=len(train), val_count=len(val),
                   best_step=best_step, best_checkpoint=str(args.out / f'candidate_step{best_step}.pt'),
                   last_checkpoint_sha256=sha256(final_path), adapter_parameters=sum(p.numel() for p in adapter.parameters()),
                   all_adapter_layers_gradient_checked=gradient_checked, backbone_frozen=True,
                   real_feature_checkpoint_roundtrip_exact=True, cached_online_deployment=online_check,
                   selected_checkpoint_deployment=selected_check,
                   seconds=time.monotonic() - started,
                   cuda_peak_bytes=torch.cuda.max_memory_allocated(), history=history, initial_train_probe=initial_probe,
                   actual_training_items_seen=items_seen,
                   protocol=protocol,
                   generation_not_yet_evaluated=True)
    write_new(args.out / 'finished.json', summary)
    print(json.dumps(dict(event='finished', complete=True, steps=args.steps, out=str(args.out))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['features', 'train'], required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model_path', type=Path, default=Path('/opt/models/MusicFM-MSD'))
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--nshard', type=int, default=1)
    parser.add_argument('--max_items', type=int, default=0)
    parser.add_argument('--targets', nargs='+', type=Path)
    parser.add_argument('--features', nargs='+', type=Path)
    parser.add_argument('--steps', type=int, default=1875)
    parser.add_argument('--eval_every', type=int, default=250)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--small_fit', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cosine_weight', type=float, default=1.0)
    parser.add_argument('--length_weight', type=float, default=1.0)
    parser.add_argument('--initialize_length_mean', action='store_true', help='Train-only log length bias initialization, no GT at inference')
    args = parser.parse_args()
    if (args.batch_size < 1 or args.steps < 1 or args.eval_every < 1 or not 0 <= args.shard < args.nshard
            or args.max_items < 0 or args.small_fit < 0 or not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.cosine_weight) or args.cosine_weight < 0
            or not math.isfinite(args.length_weight) or args.length_weight <= 0):
        parser.error('Invalid training/cache arguments')
    torch.set_num_threads(2)
    if args.stage == 'features':
        feature_cache(args)
    else:
        if not args.targets or not args.features:
            parser.error('Training requires explicit target and feature shards')
        train_adapter(args)


if __name__ == '__main__':
    main()
