"""Dense MuQ/MusicFM -> shared H3 normalized latents: bridge, CLIP, or joint training.
Single GPU / single-node torchrun; float32, frozen original MotionScript text, no DiT loss.
Feature caches are allowed only for adapter-only bridge warmup, never encoder finetuning.
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from music_encoder_adapter import AUDIO_ROOT, DEFAULT_PATHS, DenseMusicEncoder, MusicToH3
from native_audio_full_clip import clip_loss, cpu_state, finite_guard, sha256

SAMPLE_RATE, CLIP_SAMPLES, LATENT_FRAMES = 32000, 162000, 203


def load_audio(sid, audio_root=AUDIO_ROOT, load_samples=True):
    import soundfile as sf
    import torchaudio
    with sf.SoundFile(str(Path(audio_root) / (sid + '.wav'))) as stream:
        sr = stream.samplerate
        if sr <= 0 or stream.channels not in (1, 2) or stream.frames < 1:
            raise ValueError(f'Invalid source audio: {sid}')
        effective_samples = min(CLIP_SAMPLES, math.ceil(min(stream.frames, math.ceil(5.0625 * sr)) * SAMPLE_RATE / sr))
        effective_frames = min(LATENT_FRAMES, math.ceil(effective_samples / 800))
        if not load_samples:
            return torch.empty(0), effective_frames
        audio = stream.read(math.ceil(5.0625 * sr), dtype='float32', always_2d=True)
    waveform = torch.from_numpy(audio.T.copy())
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    if not torch.isfinite(waveform).all():
        raise ValueError(f'Non-finite audio: {sid}')
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    waveform = waveform[:, :CLIP_SAMPLES]
    return F.pad(waveform, (0, CLIP_SAMPLES - waveform.shape[-1])), effective_frames


class MusicPairs(Dataset):
    def __init__(self, ids, text, teacher=None, features=None, audio_root=AUDIO_ROOT):
        self.ids, self.text, self.teacher, self.features = ids, text, teacher, features
        self.audio_root = audio_root

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sid = self.ids[index]
        waveform, frames = load_audio(sid, self.audio_root, load_samples=self.features is None)
        source = waveform if self.features is None else self.features[sid].float()
        teacher = torch.empty(0) if self.teacher is None else self.teacher[sid].float()
        return source, self.text[sid], teacher, frames


class TrainingForward(nn.Module):
    def __init__(self, model, stage, cached=False):
        super().__init__()
        self.model, self.stage, self.cached = model, stage, cached

    def forward(self, source, effective_frames=None):
        if self.cached:
            batch = len(source)
            latent = self.model.adapter(source.flatten(0, 1), LATENT_FRAMES,
                                        self.model.backbone.frame_hz, self.model.backbone.frame_offset)
            latent = latent.reshape(batch, 2, 32, LATENT_FRAMES)
            return None, latent
        if self.stage == 'bridge':
            return None, self.model.encode_stereo(source)
        return self.model(source, return_latent=True, effective_frames=effective_frames)


def bridge_terms(latent, teacher, frames):
    if latent.shape != teacher.shape or latent.shape[1:] != (2, 32, LATENT_FRAMES):
        raise ValueError('Expected matching normalized [B,2,32,203] latents')
    if torch.any((frames < 1) | (frames > LATENT_FRAMES)):
        raise ValueError('Invalid effective audio frame count')
    mask = (torch.arange(LATENT_FRAMES, device=latent.device)[None, :] < frames[:, None])[:, None, None, :]
    denominator = frames.float() * 64
    mse = ((latent - teacher).square() * mask).sum((1, 2, 3)) / denominator
    energy = (teacher.square() * mask).sum((1, 2, 3)) / denominator
    prediction_energy = (latent.square() * mask).sum((1, 2, 3)) / denominator
    return mse, energy, prediction_energy


def retrieval_metrics(audio, text):
    scores = audio.float() @ text.float().T
    correct = torch.arange(len(audio), device=audio.device)[:, None]
    metrics = {}
    for direction, matrix in (('a2t', scores), ('t2a', scores.T)):
        ranking = matrix.argsort(dim=1, descending=True)
        for k in (1, 5, 10):
            metrics[f'val_{direction}_R@{k}'] = (ranking[:, :k] == correct).any(1).float().mean().item()
    for k in (1, 5, 10):
        metrics[f'val_R@{k}'] = (metrics[f'val_a2t_R@{k}'] + metrics[f'val_t2a_R@{k}']) / 2
    return metrics


@torch.no_grad()
def evaluate(forward, loader, device, baseline_mse=None):
    forward.eval()
    audio, text, errors, energies, predictions = [], [], [], [], []
    for source, target, teacher, frames in loader:
        embedding, latent = forward(source.to(device, non_blocking=True), frames.to(device))
        finite_guard([('latent', latent)], device, 'validation latent', synchronize=False)
        if embedding is not None:
            finite_guard([('embedding', embedding)], device, 'validation embedding', synchronize=False)
            audio.append(embedding)
            text.append(target.to(device))
        if teacher.numel():
            terms = bridge_terms(latent, teacher.to(device), frames.to(device))
            finite_guard(zip(('MSE', 'teacher energy', 'prediction energy'), terms), device, 'validation bridge', synchronize=False)
            errors.append(terms[0]); energies.append(terms[1]); predictions.append(terms[2])
    metrics = retrieval_metrics(torch.cat(audio), torch.cat(text)) if audio else {}
    if errors:
        mse, energy, prediction = [torch.cat(values).mean().item() for values in (errors, energies, predictions)]
        metrics.update(val_bridge_MSE=mse, val_teacher_energy=energy,
                       val_bridge_MSE_over_teacher_energy=mse / max(energy, 1e-12),
                       val_latent_energy_ratio=prediction / max(energy, 1e-12))
        if baseline_mse is not None:
            metrics['val_bridge_relative_baseline'] = mse / max(baseline_mse, 1e-12)
    return metrics


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_inputs(args):
    with open(args.caps) as stream:
        caps = json.load(stream)
    text = torch.load(args.text, map_location='cpu', weights_only=False)
    if not isinstance(caps, dict) or len(caps) != 2000 or not isinstance(text, dict) or set(caps) != set(text):
        raise ValueError('Require exactly 2000 selected IDs and matching frozen text vectors')
    if any(not isinstance(sid, str) or '/' in sid or '\\' in sid or sid in ('.', '..') for sid in caps):
        raise ValueError('Invalid selected clip ID')
    ids = sorted(caps)
    random.Random(42).shuffle(ids)
    split = dict(train=ids[200:], val=ids[:200], seed=42)
    ordered = split['train'] + split['val']
    if any(not isinstance(text[sid], torch.Tensor) or text[sid].shape != (5120,) for sid in ids):
        raise ValueError('Text cache must contain Tensor[5120]')
    vectors = torch.stack([text[sid].detach().float() for sid in ordered])
    if not torch.isfinite(vectors).all():
        raise ValueError('Non-finite frozen text')
    mean, std = vectors[:1800].mean(0), vectors[:1800].std(0).clamp_min(1e-4)
    text = dict(zip(ordered, F.normalize((vectors - mean) / std, dim=-1)))
    teacher, features, metadata = None, None, {}
    for label, path, shape in (('latents', args.teacher, (2, 32, LATENT_FRAMES)),
                               ('features', args.features, None)):
        if not path:
            continue
        data = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(data, dict):
            raise ValueError(f'{label}: expected a cache dictionary')
        values, meta = data.get(label), data.get('metadata')
        if not isinstance(values, dict) or set(values) != set(ids) or not isinstance(meta, dict):
            raise ValueError(f'{label}: require matching 2000 IDs and metadata')
        for sid, value in values.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(f'Expected tensor {label}: {sid}')
            valid_shape = value.shape == shape if shape else (value.ndim == 3 and value.shape[0] == 2 and value.shape[1] > 0 and value.shape[2] == 1024)
            if not valid_shape or not torch.isfinite(value).all():
                raise ValueError(f'Invalid {label}: {sid}')
        if label == 'features':
            expected = dict(backend=args.backend, sample_rate=24000, frame_hz=25.0, hidden_size=1024)
            if any(meta.get(key) != value for key, value in expected.items()):
                raise ValueError('Feature metadata disagrees with the official dense backbone')
            if Path(meta.get('model_path', '')).resolve() != Path(args.model_path).resolve():
                raise ValueError('Feature cache and initialized backbone model_path differ')
            if len({tuple(value.shape) for value in values.values()}) != 1:
                raise ValueError('Fixed-duration feature cache must have a consistent sequence length')
            features = values
        else:
            expected = dict(normalized=True, sample_rate=32000, hop_length=800)
            if any(key in meta and meta[key] != value for key, value in expected.items()):
                raise ValueError('Teacher metadata contradicts normalized native H3 latents')
            teacher = values
        metadata[label] = meta
    return split, text, mean, std, teacher, features, metadata


def initialize_adapter(model, path, split, backend):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('split') != split or checkpoint.get('backend') != backend:
        raise ValueError('Warmup initialization requires identical split and backend')
    if not isinstance(checkpoint.get('adapter'), dict):
        raise ValueError('--init loads only a warmup adapter checkpoint, not a full model/resume')
    model.adapter.load_state_dict(checkpoint['adapter'], strict=True)


def parameter_counts(model):
    return {name: dict(total=sum(p.numel() for p in module.parameters()),
                       trainable=sum(p.numel() for p in module.parameters() if p.requires_grad))
            for name, module in (('backbone', model.backbone), ('adapter', model.adapter), ('audio_head', model.audio_head))}


def gradient_report(model, device, detailed=True):
    tensors = [(name, p.grad) for name, p in model.named_parameters() if p.requires_grad and p.grad is not None]
    if not tensors:
        raise RuntimeError('No trainable gradients')
    finite_guard(tensors, device, 'gradients')
    if not detailed:
        return {}
    report = {}
    for name, module in (('backbone', model.backbone), ('adapter', model.adapter), ('audio_head', model.audio_head)):
        gradients = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
        if any(p.requires_grad for p in module.parameters()) and not gradients:
            raise RuntimeError(f'No gradients reached trainable {name}')
        report[name] = dict(grad_norm=math.sqrt(sum(g.float().square().sum().item() for g in gradients)),
                            parameters_with_grad=sum(p.numel() for p in module.parameters() if p.grad is not None),
                            missing_grad_tensors=sum(p.requires_grad and p.grad is None for p in module.parameters()))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('bridge', 'clip', 'bridge_clip'))
    parser.add_argument('--backend', required=True, choices=('muq', 'musicfm'))
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--caps', required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--teacher', default=None)
    parser.add_argument('--features', default=None)
    parser.add_argument('--init', default=None, help='Load adapter only from same-split bridge warmup')
    parser.add_argument('--out', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=2, help='Per-rank batch size')
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--adapter_lr', type=float, default=1e-4)
    parser.add_argument('--head_lr', type=float, default=1e-4)
    parser.add_argument('--bridge_weight', type=float, default=1.0)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        parser.error('Use a fresh output directory; existing training artifacts will not be overwritten')
    if args.stage == 'clip' and args.teacher is not None:
        parser.error('Strict CLIP must not declare or load a teacher path')
    if args.stage != 'clip' and not args.teacher:
        parser.error('bridge and bridge_clip require --teacher')
    if args.features and args.stage != 'bridge':
        parser.error('--features is adapter warmup only; encoder training must load raw audio')
    numeric = (args.epochs, args.batch_size, args.encoder_lr, args.adapter_lr, args.head_lr)
    if any(not math.isfinite(value) or value <= 0 for value in numeric):
        parser.error('Epochs, batch size and learning rates must be finite and positive')
    if not math.isfinite(args.bridge_weight) or args.bridge_weight < 0:
        parser.error('Bridge weight must be finite and nonnegative')
    args.model_path = args.model_path or DEFAULT_PATHS[args.backend]
    world, rank, local_rank = [int(os.environ.get(key, default)) for key, default in
                               (('WORLD_SIZE', '1'), ('RANK', '0'), ('LOCAL_RANK', '0'))]
    if world * args.batch_size < 2 and args.stage != 'bridge':
        parser.error('InfoNCE requires at least two global paired items')
    if int(os.environ.get('LOCAL_WORLD_SIZE', str(world))) != world:
        parser.error('Only single-node DDP is supported')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(42)
    random.seed(42)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl', device_id=device)
    split, text, mean, std, teacher, features, metadata = load_inputs(args)
    model = MusicToH3(DenseMusicEncoder(args.backend, args.model_path))
    metadata['backbone'] = model.backbone.metadata()
    if args.init:
        initialize_adapter(model, args.init, split, args.backend)
    if args.stage == 'bridge':
        model.backbone.requires_grad_(False)
        model.audio_head.requires_grad_(False)
    model.to(device=device, dtype=torch.float32)
    finite_guard(model.named_parameters(), device, 'initial weights')
    forward = TrainingForward(model, args.stage, cached=features is not None)
    train_model = DDP(forward, device_ids=[local_rank], broadcast_buffers=False,
                      find_unused_parameters=True) if world > 1 else forward
    groups = [dict(params=[p for p in module.parameters() if p.requires_grad], lr=lr,
                   weight_decay=0.0 if name == 'backbone' else 0.01)
              for name, module, lr in (('backbone', model.backbone, args.encoder_lr),
                                      ('adapter', model.adapter, args.adapter_lr), ('audio_head', model.audio_head, args.head_lr))
              if any(p.requires_grad for p in module.parameters())]
    optimizer = torch.optim.AdamW(groups)
    dataset = MusicPairs(split['train'], text, teacher, features)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=42, drop_last=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True,
                        num_workers=2, pin_memory=True, persistent_workers=True)
    if not len(loader):
        raise ValueError('Training split smaller than global batch')
    val_loader = DataLoader(MusicPairs(split['val'], text, teacher, features), batch_size=args.batch_size,
                            num_workers=2, pin_memory=True) if rank == 0 else None
    counts, baseline, best_score, best_metrics = parameter_counts(model), {}, -math.inf, None
    out = Path(args.out)
    hashes = {key: sha256(path) for key, path in (('caps', args.caps), ('text', args.text),
              ('teacher', args.teacher), ('features', args.features), ('init', args.init)) if path} if rank == 0 else {}
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        baseline = evaluate(forward, val_loader, device)
        print(json.dumps(dict(stage=args.stage, parameters=counts, baseline=baseline, dtype='float32',
                              bn_caveat='bridge backbone eval; joint/CLIP train BN uses per-rank stereo batch; validation uses running stats, not SyncBN',
                              input_samples=CLIP_SAMPLES, latent_frames=LATENT_FRAMES)), flush=True)
    if world > 1:
        dist.barrier()
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch - 1)
        train_model.train()
        if args.stage == 'bridge':
            model.backbone.eval()
        sums = torch.zeros(3, device=device)
        for index, (source, target, teacher_batch, frames) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            embedding, latent = train_model(source.to(device, non_blocking=True), frames.to(device))
            finite_guard([('latent', latent)] + ([('embedding', embedding)] if embedding is not None else []), device, 'forward')
            lclip = clip_loss(embedding, target.to(device)) if args.stage != 'bridge' else latent.new_zeros(())
            lbridge = bridge_terms(latent, teacher_batch.to(device), frames.to(device))[0].mean() if args.stage != 'clip' else latent.new_zeros(())
            loss = lbridge if args.stage == 'bridge' else lclip + args.bridge_weight * lbridge
            finite_guard([('loss', loss)], device, 'loss')
            loss.backward()
            gradients = gradient_report(model, device, detailed=index == 0 or (index + 1) % 20 == 0)
            norm = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
            optimizer.step()
            sums += torch.stack([loss.detach(), lclip.detach(), lbridge.detach()])
            if index == 0 or (index + 1) % 20 == 0:
                finite_guard(((name, p) for name, p in model.named_parameters() if p.requires_grad), device, 'updated weights')
                if rank == 0:
                    print(json.dumps(dict(epoch=epoch, batch=index + 1, loss=loss.item(), clip_loss=lclip.item(),
                                          bridge_loss=lbridge.item(), grad_norm=norm.item(), gradients=gradients,
                                          cuda_peak_gib=torch.cuda.max_memory_allocated(device) / 2**30)), flush=True)
        if world > 1:
            dist.all_reduce(sums)
        record = dict(epoch=epoch, stage=args.stage, **dict(zip(('train_loss', 'train_clip_loss', 'train_bridge_MSE'),
                                                              (sums / (world * len(loader))).tolist())))
        if epoch % 5 == 0 or epoch == args.epochs:
            if world > 1:
                dist.barrier()
            if rank == 0:
                metrics = evaluate(forward, val_loader, device, baseline.get('val_bridge_MSE'))
                record.update(metrics)
                checkpoint = dict(adapter=cpu_state(model.adapter), backend=args.backend, model_path=args.model_path,
                                  config=vars(args), split=split, epoch=epoch, metrics=metrics, baseline=baseline,
                                  input_sha256=hashes, input_metadata=metadata, text_mean=mean, text_std=std,
                                  stage=args.stage, interface=dict(sample_rate=32000, input_samples=162000,
                                  latent_channels=32, latent_frames=203, normalized=True, shared_clip_generator_latent=True))
                if args.stage != 'bridge':
                    checkpoint['model'] = cpu_state(model)
                atomic_save(checkpoint, out / 'last.pt')
                score = -metrics['val_bridge_MSE'] if args.stage == 'bridge' else metrics['val_R@10']
                if score > best_score:
                    best_score, best_metrics = score, metrics
                    atomic_save(checkpoint, out / 'best.pt')
                del checkpoint
            if world > 1:
                dist.barrier()
        if rank == 0:
            with (out / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            print(json.dumps(record, allow_nan=False), flush=True)
    if rank == 0:
        summary = dict(config=vars(args), split=split, parameters=counts, baseline=baseline,
                       best_metrics=best_metrics, input_sha256=hashes, input_metadata=metadata,
                       checkpoint_sha256={name: sha256(out / (name + '.pt')) for name in ('last', 'best')},
                       initialization=dict(backbone='public initialization', adapter=args.init or 'random', audio_head='random'),
                       bn_caveat='per-rank BN train stats versus validation running stats; bridge backbone always eval', dtype='float32')
        with (out / 'summary.json').open('w') as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
