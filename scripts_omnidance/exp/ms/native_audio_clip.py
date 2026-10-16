"""仅音频侧CLIP的最小原生接口消融：训练audio-VAE末端mean_proj及音频侧投影。
encoder/pre_block缓存、文本向量、DiT均冻结；不加入重建或视频loss，不检索caption作生成条件。
--native_lr 0为冻结原生latent的对照；非零为只更新原生末端投影的候选。
5120维音频侧投影仅用于CLIP；推理只导出32维mean_proj，保留原生音频时间网格。
本实验不等于全音频encoder微调。检索指标只衡量表征，需另做原生H3生成/音频保真验收。
"""
import argparse
import glob
import hashlib
import json
import math
import os
import random

import torch
from torch import nn
from torch.nn import functional as F


class NativeAudioEncoder(nn.Module):
    def __init__(self, mean_proj, mean, std, text_dim):
        super().__init__()
        self.mean_proj = nn.Conv1d(32, 32, 1)
        self.mean_proj.load_state_dict(mean_proj)
        self.register_buffer('mean', mean.float().reshape(1, 1, 32, 1))
        self.register_buffer('std', std.float().reshape(1, 1, 32, 1))
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 512), nn.GELU(), nn.Linear(512, text_dim))

    def latent(self, x):
        shape = x.shape
        raw = self.mean_proj(x.flatten(0, 1)).reshape(shape)
        return (raw - self.mean) / self.std

    def forward(self, x):
        latent = self.latent(x)
        pooled = torch.cat([latent.mean(-1).flatten(1), latent.std(-1).flatten(1)], 1)
        return F.normalize(self.head(pooled), dim=-1)


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def evaluate(model, x, targets, native_baseline, caps, ids):
    model.eval()
    with torch.no_grad():
        embeddings = torch.cat([model(x[start:start + 256]) for start in range(0, len(ids), 256)])
        scores = embeddings @ targets.T
        ranking = scores.argsort(dim=1, descending=True)
        correct = torch.arange(len(ids), device=scores.device)[:, None]
        metrics = {f'R@{k}': (ranking[:, :k] == correct).any(1).float().mean().item() for k in (1, 5, 10)}
        nearest = ranking[:, 0].cpu().tolist()
        for key in ('genre', 'coarse', 'energy', 'tempo', 'quality'):
            values = [caps[sid]['dims'][key] for sid in ids]
            metrics['hit_' + key] = sum(values[i] == values[j] for i, j in enumerate(nearest)) / len(ids)
            counts = {value: values.count(value) for value in set(values)}
            metrics['random_' + key] = sum(count ** 2 for count in counts.values()) / len(ids) ** 2
        current = model.latent(x)
        difference = current - native_baseline
        metrics['native_latent_relative_rms_change'] = (difference.square().mean().sqrt() / native_baseline.square().mean().sqrt()).item()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--caches', nargs='+', required=True, help='Native feature PT shards or glob patterns')
    parser.add_argument('--text', required=True)
    parser.add_argument('--caps', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--native_lr', type=float, default=1e-5)
    parser.add_argument('--head_lr', type=float, default=3e-4)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    paths = sorted({path for pattern in args.caches for path in glob.glob(pattern)})
    if not paths:
        raise ValueError('No native feature cache shards')
    features, first = {}, None
    for path in paths:
        shard = torch.load(path, map_location='cpu', weights_only=False)
        if first is None:
            first = shard
        else:
            for key in ('weight', 'bias'):
                if not torch.equal(first['mean_proj'][key], shard['mean_proj'][key]):
                    raise ValueError('Native cache shards have different base weights')
            if not torch.equal(first['mean'], shard['mean']) or not torch.equal(first['std'], shard['std']):
                raise ValueError('Native cache shards have different normalization')
        if features.keys() & shard['features'].keys():
            raise ValueError('Duplicate IDs across native cache shards')
        features.update(shard['features'])
    targets = torch.load(args.text, map_location='cpu', weights_only=False)
    caps = json.load(open(args.caps))
    ids = sorted(set(features) & set(targets) & set(caps))
    if set(ids) != set(caps):
        raise ValueError(f'Incomplete selected dataset: {len(ids)} of {len(caps)} paired items')
    random.Random(42).shuffle(ids)
    nv = max(2, int(len(ids) * args.val_fraction))
    if len(ids) - nv < args.batch_size or args.epochs < 1:
        raise ValueError('Not enough training items for this split and batch size')
    val_ids, train_ids = ids[:nv], ids[nv:]
    order = train_ids + val_ids
    x = torch.stack([features[sid].float() for sid in order]).cuda()
    text = torch.stack([targets[sid].float() for sid in order])
    text_mean, text_std = text[:len(train_ids)].mean(0), text[:len(train_ids)].std(0).clamp_min(1e-4)
    text = F.normalize((text - text_mean) / text_std, dim=-1).cuda()
    model = NativeAudioEncoder(first['mean_proj'], first['mean'], first['std'], text.shape[1]).cuda()
    model.mean_proj.requires_grad_(args.native_lr > 0)
    log_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), device='cuda'))
    groups = [{'params': list(model.head.parameters()) + [log_scale], 'lr': args.head_lr}]
    if args.native_lr > 0:
        groups.append({'params': model.mean_proj.parameters(), 'lr': args.native_lr, 'weight_decay': 0.0})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    ntrain = len(train_ids)
    with torch.no_grad():
        baseline = model.latent(x[ntrain:]).clone()
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, 'metrics.jsonl'), 'a')
    json.dump(dict(train=train_ids, val=val_ids), open(os.path.join(args.out, 'split.json'), 'w'))
    best_score = -1.0
    best_metrics = None
    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(ntrain, device='cuda')
        losses = []
        for start in range(0, ntrain, args.batch_size):
            indices = permutation[start:start + args.batch_size]
            if len(indices) < 2:
                continue
            embeddings = model(x[indices])
            logits = log_scale.exp().clamp(max=100) * embeddings @ text[indices].T
            labels = torch.arange(len(indices), device='cuda')
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite CLIP loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise RuntimeError('Non-finite audio encoder gradient')
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            metrics = evaluate(model, x[ntrain:], text[ntrain:], baseline, caps, val_ids)
            record = dict(epoch=epoch + 1, train_loss=sum(losses) / len(losses), **metrics)
            log.write(json.dumps(record, allow_nan=False) + '\n')
            log.flush()
            print(json.dumps(record), flush=True)
            score = metrics['R@10']
            checkpoint = dict(model=cpu_state(model), native_mean_proj=cpu_state(model.mean_proj),
                              text_mean=text_mean, text_std=text_std, metrics=metrics, args=vars(args),
                              note='Audio-side CLIP; only mean_proj is installed in native H3 inference. No generation validation yet.')
            if score > best_score:
                best_score, best_metrics = score, metrics
                torch.save(checkpoint, os.path.join(args.out, 'best.pt'))
            if epoch == args.epochs - 1:
                torch.save(checkpoint, os.path.join(args.out, 'last.pt'))
    log.close()
    checkpoint = torch.load(os.path.join(args.out, 'best.pt'), map_location='cpu', weights_only=False)
    if not all(torch.isfinite(value).all() for value in checkpoint['model'].values()):
        raise RuntimeError('Non-finite saved weights')
    with open(os.path.join(args.out, 'best.pt'), 'rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    json.dump(dict(best_metrics=best_metrics, sha256=digest, args=vars(args), stage='representation_only'),
              open(os.path.join(args.out, 'summary.json'), 'w'), indent=2, allow_nan=False)
    print('DONE', digest, flush=True)


if __name__ == '__main__':
    main()
