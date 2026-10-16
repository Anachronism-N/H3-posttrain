"""冻结音乐特征的公平线性 probe；仅训练 LayerNorm(4096)+Linear(4096,5120,bias=False)。
输入 features:{sid:Tensor[2,T,1024]}；同2000ID、seed42、1800训练/200验证。
时间mean/std保留两个声道，文本与音频均只用训练统计标准化；文本与encoder不训练。
检索仅为 frozen_feature_probe，不证明音乐条件生成能力；线性不代表参数量小。
"""
import argparse
import hashlib
import json
import math
import os
import random

import torch
from torch import nn
from torch.nn import functional as F


FEATURE_DIM, POOL_DIM, TEXT_DIM = 1024, 4096, 5120
SEED, TRAIN_ITEMS, VAL_ITEMS = 42, 1800, 200
KS = (1, 5, 10)


def finite_guard(named_tensors, stage):
    bad = [name for name, tensor in named_tensors
           if tensor is None or not torch.isfinite(tensor).all().item()]
    if bad:
        raise RuntimeError(f'{stage}: non-finite/missing tensors: {bad}')


def pool_features(feature):
    """[C=2,T,D=1024] -> [mean(channel0/1),std(channel0/1)], population std."""
    if not isinstance(feature, torch.Tensor) or feature.ndim != 3:
        raise ValueError('Each feature must be a Tensor[2,T,1024]')
    if feature.shape[0] != 2 or feature.shape[1] < 1 or feature.shape[2] != FEATURE_DIM:
        raise ValueError(f'Expected [2,T>=1,1024], got {tuple(feature.shape)}')
    feature = feature.detach().float()
    finite_guard([('feature', feature)], 'feature pooling')
    return torch.cat((feature.mean(1).flatten(), feature.std(1, unbiased=False).flatten()))


class LinearAudioHead(nn.Sequential):
    def __init__(self):
        super().__init__(nn.LayerNorm(POOL_DIM), nn.Linear(POOL_DIM, TEXT_DIM, bias=False))

    def forward(self, pooled):
        return F.normalize(super().forward(pooled), dim=-1)


def clip_loss(audio, text, temperature=0.07):
    scores = audio.float() @ text.float().T / temperature
    labels = torch.arange(len(audio), device=audio.device)
    return (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)) / 2


def retrieval_metrics(scores, caps, ids, temperature=0.07):
    """双方均以全部传入ID作候选；条件检索仅保留与query同舞种的候选。"""
    n = len(ids)
    if n < 2 or scores.shape != (n, n):
        raise ValueError('Retrieval requires a square score matrix and >=2 paired IDs')
    finite_guard([('scores', scores)], 'retrieval')
    correct = torch.arange(n, device=scores.device)[:, None]
    logits = scores.float() / temperature
    metrics = dict(n_queries=n, n_candidates=n,
                   loss=((F.cross_entropy(logits, correct[:, 0]) +
                          F.cross_entropy(logits.T, correct[:, 0])) / 2).item())
    rankings = {}
    for direction, matrix in (('a2t', scores), ('t2a', scores.T)):
        ranking = matrix.argsort(dim=1, descending=True)
        rankings[direction] = ranking
        for k in KS:
            metrics[f'{direction}_R@{k}'] = (ranking[:, :k] == correct).any(1).float().mean().item()
    for k in KS:
        metrics[f'random_R@{k}'] = min(k, n) / n
    conditional = {}
    for key in ('coarse', 'genre'):
        values = [caps[sid]['dims'][key] for sid in ids]
        groups = {}
        for index, value in enumerate(values):
            groups.setdefault(value, []).append(index)
        for direction, ranking in rankings.items():
            nearest = ranking[:, 0].tolist()
            metrics[f'{direction}_hit_{key}'] = sum(values[i] == values[j] for i, j in enumerate(nearest)) / n
        metrics[f'random_{key}'] = sum(len(indices) ** 2 for indices in groups.values()) / n ** 2
        eligible = [indices for indices in groups.values() if len(indices) >= 2]
        evaluated_n = sum(map(len, eligible))
        result = dict(n=evaluated_n, skipped_singletons=n - evaluated_n, n_classes=len(eligible))
        if evaluated_n:
            sizes = [len(indices) for indices in eligible]
            result.update(candidate_count_min=min(sizes), candidate_count_max=max(sizes),
                          candidate_count_query_mean=sum(size ** 2 for size in sizes) / evaluated_n)
            hits = {f'{direction}_R@{k}': 0 for direction in ('a2t', 't2a') for k in KS}
            for indices in eligible:
                indices = torch.tensor(indices, device=scores.device)
                matrix = scores.index_select(0, indices).index_select(1, indices)
                group_correct = torch.arange(len(indices), device=scores.device)[:, None]
                for direction, candidate_scores in (('a2t', matrix), ('t2a', matrix.T)):
                    ranking = candidate_scores.argsort(dim=1, descending=True)
                    for k in KS:
                        hits[f'{direction}_R@{k}'] += (ranking[:, :k] == group_correct).any(1).sum().item()
            result.update({key: value / evaluated_n for key, value in hits.items()})
            result.update({f'random_R@{k}': sum(min(k, size) for size in sizes) / evaluated_n for k in KS})
        else:
            result.update({f'{direction}_R@{k}': None for direction in ('a2t', 't2a') for k in KS})
            result.update({f'random_R@{k}': None for k in KS})
        conditional[key] = result
    metrics['same_category_ranking'] = conditional
    return metrics


@torch.no_grad()
def evaluate(model, features, targets, caps, ids, temperature, batch_size):
    model.eval()
    audio = torch.cat([model(features[start:start + batch_size])
                       for start in range(0, len(ids), batch_size)])
    finite_guard([('audio', audio), ('text', targets)], 'evaluation embeddings')
    return retrieval_metrics(audio @ targets.T, caps, ids, temperature)


def cpu_state(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def prepare_inputs(args, device):
    cache = torch.load(args.features, map_location='cpu', weights_only=True)
    targets = torch.load(args.text, map_location='cpu', weights_only=True)
    with open(args.caps) as stream:
        caps = json.load(stream)
    if not isinstance(cache, dict) or not isinstance(cache.get('features'), dict) or not isinstance(cache.get('metadata'), dict):
        raise ValueError('Feature cache must contain features and metadata dictionaries')
    features = cache['features']
    if not isinstance(targets, dict) or not isinstance(caps, dict):
        raise ValueError('Text and caption files must be ID dictionaries')
    if len(caps) != TRAIN_ITEMS + VAL_ITEMS or set(caps) != set(features) or set(caps) != set(targets):
        raise ValueError('Exactly the same 2000 IDs are required in features, text and captions; no intersection-only matching')
    for sid in caps:
        if not isinstance(sid, str) or '/' in sid or '\\' in sid or sid in ('.', '..'):
            raise ValueError('Invalid selected clip ID')
        caption = caps[sid]
        if not isinstance(caption, dict) or not isinstance(caption.get('dims'), dict):
            raise ValueError(f'{sid}: missing caption dims')
        if any(not isinstance(caption['dims'].get(key), str) or not caption['dims'][key] for key in ('coarse', 'genre')):
            raise ValueError(f'{sid}: coarse and genre must be nonempty strings')
        if not isinstance(targets[sid], torch.Tensor) or targets[sid].shape != (TEXT_DIM,):
            raise ValueError(f'{sid}: text must be a Tensor[5120]')
        if not isinstance(features[sid], torch.Tensor):
            raise ValueError(f'{sid}: feature must be a Tensor[2,T,1024]')
    ids = sorted(caps)
    random.Random(SEED).shuffle(ids)
    val_ids, train_ids = ids[:VAL_ITEMS], ids[VAL_ITEMS:]
    order = train_ids + val_ids
    pooled = torch.stack([pool_features(features[sid].to(device)) for sid in order])
    text = torch.stack([targets[sid].detach().to(device=device, dtype=torch.float32) for sid in order])
    finite_guard([('pooled', pooled), ('text', text)], 'input cache')
    feature_mean = pooled[:TRAIN_ITEMS].mean(0)
    feature_std = pooled[:TRAIN_ITEMS].std(0).clamp_min(1e-4)
    text_mean = text[:TRAIN_ITEMS].mean(0)
    text_std = text[:TRAIN_ITEMS].std(0).clamp_min(1e-4)
    pooled = (pooled - feature_mean) / feature_std
    text = (text - text_mean) / text_std
    if (text.norm(dim=-1) == 0).any().item():
        raise ValueError('A standardized text target has zero norm')
    text = F.normalize(text, dim=-1)
    finite_guard([('features', pooled), ('targets', text), ('feature_mean', feature_mean),
                  ('feature_std', feature_std), ('text_mean', text_mean), ('text_std', text_std)], 'precomputed normalization')
    normalization = dict(feature_mean=feature_mean.cpu(), feature_std=feature_std.cpu(),
                         text_mean=text_mean.cpu(), text_std=text_std.cpu())
    split = dict(train=train_ids, val=val_ids, train_same200=train_ids[:VAL_ITEMS], seed=SEED)
    return pooled, text, caps, split, normalization, cache['metadata']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', required=True, help='Frozen {features:{sid:Tensor[2,T,1024]},metadata:{...}} PT')
    parser.add_argument('--text', required=True, help='Frozen text_orig_2k.pt: {sid:Tensor[5120]}')
    parser.add_argument('--caps', required=True, help='captions_orig_2k.json: {sid:{dims:{coarse,genre,...}}}')
    parser.add_argument('--out', required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.07, help='Constant temperature; never trained')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    args = parser.parse_args()
    if args.epochs < 1 or not 2 <= args.batch_size <= TRAIN_ITEMS:
        raise ValueError('epochs >=1 and 2 <= batch_size <=1800 are required')
    if not all(math.isfinite(value) and value > 0 for value in (args.lr, args.temperature)):
        raise ValueError('lr and temperature must be finite and positive')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError('weight_decay must be finite and nonnegative')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This frozen-feature probe is single-GPU only; do not use multi-rank torchrun')
    if not torch.cuda.is_available():
        raise RuntimeError('Actual probe training requires one CUDA GPU')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
    features, targets, caps, split, normalization, metadata = prepare_inputs(args, device)
    model = LinearAudioHead().to(device)
    finite_guard(model.named_parameters(), 'initial head weights')
    if features.requires_grad or targets.requires_grad:
        raise RuntimeError('Cached features and text targets must be frozen')
    parameter_counts = dict(linear=sum(p.numel() for p in model[1].parameters()),
                            layer_norm=sum(p.numel() for p in model[0].parameters()),
                            total=sum(p.numel() for p in model.parameters()), encoder=0, text=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    os.makedirs(args.out, exist_ok=True)
    if any(os.path.exists(os.path.join(args.out, name)) for name in ('metrics.jsonl', 'split.json', 'best.pt', 'last.pt', 'summary.json')):
        raise FileExistsError('Use a fresh output directory; existing probe outputs will not be overwritten')
    with open(os.path.join(args.out, 'split.json'), 'w') as stream:
        json.dump(split, stream, indent=2)
    input_hashes = {name: sha256(path) for name, path in (('features', args.features), ('text', args.text), ('caps', args.caps))}
    best_score, best_metrics, best_epoch, last_metrics = -1.0, None, None, None
    with open(os.path.join(args.out, 'metrics.jsonl'), 'w') as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            permutation = torch.randperm(TRAIN_ITEMS, device=device)
            loss_sum, seen = 0.0, 0
            for start in range(0, TRAIN_ITEMS, args.batch_size):
                indices = permutation[start:start + args.batch_size]
                if len(indices) < 2:
                    continue
                optimizer.zero_grad(set_to_none=True)
                embedding = model(features[indices])
                loss = clip_loss(embedding, targets[indices], args.temperature)
                finite_guard([('embedding', embedding), ('loss', loss)], 'training loss')
                loss.backward()
                finite_guard(((name, parameter.grad) for name, parameter in model.named_parameters()), 'head gradients')
                optimizer.step()
                finite_guard(model.named_parameters(), 'updated head weights')
                loss_sum += loss.item() * len(indices)
                seen += len(indices)
            record = dict(epoch=epoch, train_minibatch_loss=loss_sum / seen,
                          train_items_seen=seen, stage='frozen_feature_probe')
            if epoch % 5 == 0 or epoch == args.epochs:
                metrics = dict(
                    val=evaluate(model, features[TRAIN_ITEMS:], targets[TRAIN_ITEMS:], caps,
                                 split['val'], args.temperature, args.batch_size),
                    train_same200=evaluate(model, features[:VAL_ITEMS], targets[:VAL_ITEMS], caps,
                                           split['train_same200'], args.temperature, args.batch_size))
                record.update(metrics)
                checkpoint = dict(head=cpu_state(model), normalization=normalization, split=split,
                                  epoch=epoch, metrics=metrics, args=vars(args), feature_metadata=metadata,
                                  input_sha256=input_hashes, trainable_parameters=parameter_counts,
                                  stage='frozen_feature_probe', pooling='time_mean_std_per_channel_flatten_4096',
                                  time_std_correction=0, normalization_std_correction=1)
                finite_guard(checkpoint['head'].items(), 'checkpoint head weights')
                torch.save(checkpoint, os.path.join(args.out, 'last.pt'))
                last_metrics = metrics
                score = (metrics['val']['a2t_R@10'] + metrics['val']['t2a_R@10']) / 2
                if score > best_score:
                    best_score, best_metrics, best_epoch = score, metrics, epoch
                    torch.save(checkpoint, os.path.join(args.out, 'best.pt'))
            log.write(json.dumps(record, allow_nan=False) + '\n')
            log.flush()
            print(json.dumps(record, allow_nan=False), flush=True)
    for name in ('best', 'last'):
        saved = torch.load(os.path.join(args.out, name + '.pt'), map_location='cpu', weights_only=True)
        finite_guard(saved['head'].items(), f'saved {name} head weights')
    summary = dict(stage='frozen_feature_probe', best_epoch=best_epoch, best_score=best_score,
                   best_selection='mean(val a2t_R@10, val t2a_R@10)', best_metrics=best_metrics,
                   last_metrics=last_metrics, args=vars(args), trainable_parameters=parameter_counts,
                   sha256={**input_hashes, **{name: sha256(os.path.join(args.out, name + '.pt')) for name in ('best', 'last')}},
                   pooling='time_mean_std_per_channel_flatten_4096', time_std_correction=0,
                   normalization='per-dimension train-only mean/std, std correction=1, min=1e-4; text and head output L2',
                   split_seed=SEED, training_items=TRAIN_ITEMS, validation_items=VAL_ITEMS,
                   encoder_trainable=False, text_trainable=False,
                   scope='Frozen-feature retrieval diagnostic only; not evidence of music-conditioned generation quality.',
                   capacity_note='Weak means LN+linear expression, not few parameters: 4096-to-5120 linear has ~21M weights; not smaller than the earlier 128-dimensional head.',
                   candidate_bias_note='train_same200 is the fixed first 200 training IDs; val also has 200 candidates. Do not compare against 1800-candidate train retrieval. Same-category ranking uses variable group sizes, skips singleton classes, and reports eligible n plus size-adjusted random baselines.')
    with open(os.path.join(args.out, 'summary.json'), 'w') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print('DONE frozen_feature_probe', flush=True)


if __name__ == '__main__':
    main()
