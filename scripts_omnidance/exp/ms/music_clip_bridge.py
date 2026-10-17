"""Dense MuQ/MusicFM -> shared H3 normalized latents: bridge, CLIP, or joint training.
Single GPU / single-node torchrun; float32, frozen original MotionScript text, no DiT loss.
Feature caches are allowed only for adapter-only bridge warmup, never encoder finetuning.
"""
import argparse
import hashlib
import json
import math
import os
import random
import time
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
    metrics = {'val_candidates': len(audio)}
    for k in (1, 5, 10):
        metrics[f'val_random_R@{k}'] = min(k, len(audio)) / len(audio)
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
    metrics = retrieval_metrics(torch.cat(audio), torch.cat(text)) if audio else {
        'val_candidates': len(loader.dataset),
        **{f'val_random_R@{k}': min(k, len(loader.dataset)) / len(loader.dataset) for k in (1, 5, 10)}}
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


INTERFACE = dict(sample_rate=32000, input_samples=162000, latent_channels=32,
                 latent_frames=203, normalized=True, shared_clip_generator_latent=True)


def ids_fingerprint(ids):
    return hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def validate_ids(ids, label, minimum=2):
    if not isinstance(ids, list) or len(ids) < minimum:
        raise ValueError(f'{label}: expected at least {minimum} clip IDs in a JSON string list')
    if any(not isinstance(sid, str) or not sid or '/' in sid or '\\' in sid or sid in ('.', '..') for sid in ids):
        raise ValueError(f'{label}: invalid clip ID')
    if len(set(ids)) != len(ids):
        raise ValueError(f'{label}: duplicate clip IDs')


def validate_split(split):
    if not isinstance(split, dict):
        raise ValueError('Split must be a JSON object with train and val lists')
    for label in ('train', 'val'):
        validate_ids(split.get(label), label)
    if set(split['train']) & set(split['val']):
        raise ValueError('Train and validation IDs intersect')


def load_inputs(args):
    with open(args.caps) as stream:
        caps = json.load(stream)
    text_cache = torch.load(args.text, map_location='cpu', weights_only=False)
    if not isinstance(caps, dict) or not isinstance(text_cache, dict):
        raise ValueError('Require caption and frozen text cache dictionaries')
    validate_ids(list(caps), 'caps')
    split_path = getattr(args, 'split', None)
    if split_path:
        with open(split_path) as stream:
            split = json.load(stream)
        validate_split(split)
    else:
        if len(caps) != 2000 or set(caps) != set(text_cache):
            raise ValueError('Legacy mode requires exactly 2000 selected IDs and matching frozen text vectors')
        ids = sorted(caps)
        random.Random(42).shuffle(ids)
        split = dict(train=ids[200:], val=ids[:200], seed=42)
    ordered = split['train'] + split['val']
    normalization_path = getattr(args, 'normalization_ids', None)
    if normalization_path:
        with open(normalization_path) as stream:
            normalization_ids = json.load(stream)
        validate_ids(normalization_ids, 'normalization_ids')
    else:
        normalization_ids = list(split['train'])
    if set(normalization_ids) & set(split['val']):
        raise ValueError('Normalization IDs intersect current validation IDs')
    required = set(ordered) | set(normalization_ids)
    if not required <= set(caps) or not required <= set(text_cache):
        raise ValueError('Caption/text caches must cover all split and normalization IDs')
    if any(not isinstance(text_cache[sid], torch.Tensor) or text_cache[sid].shape != (5120,) for sid in required):
        raise ValueError('Text cache must contain Tensor[5120]')
    vectors = torch.stack([text_cache[sid].detach().float() for sid in ordered])
    statistics_vectors = torch.stack([text_cache[sid].detach().float() for sid in normalization_ids])
    if not torch.isfinite(vectors).all() or not torch.isfinite(statistics_vectors).all():
        raise ValueError('Non-finite frozen text')
    mean = statistics_vectors.mean(0)
    std = statistics_vectors.std(0).clamp_min(1e-4)
    stats_sha256 = hashlib.sha256(mean.contiguous().numpy().tobytes() + std.contiguous().numpy().tobytes()).hexdigest()
    text = dict(zip(ordered, F.normalize((vectors - mean) / std, dim=-1)))
    teacher, features = None, None
    metadata = dict(subset=dict(train_count=len(split['train']), val_count=len(split['val']),
                               train_sha256=ids_fingerprint(split['train']), val_sha256=ids_fingerprint(split['val']),
                               split_source=split_path or 'legacy_seed42_1800_200'),
                    cache_id_counts=dict(caps=len(caps), text=len(text_cache)),
                    normalization=dict(ids=normalization_ids, ids_sha256=ids_fingerprint(normalization_ids),
                                       stats_sha256=stats_sha256, count=len(normalization_ids),
                                       source=normalization_path or 'current_train'))
    for label, path, shape in (('latents', args.teacher, (2, 32, LATENT_FRAMES)),
                               ('features', args.features, None)):
        if not path:
            continue
        data = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(data, dict):
            raise ValueError(f'{label}: expected a cache dictionary')
        values, meta = data.get(label), data.get('metadata')
        if not isinstance(values, dict) or not isinstance(meta, dict):
            raise ValueError(f'{label}: require cache IDs and metadata')
        expected_path = 'sourcewav->32k->24k' if label == 'features' else 'sourcewav->32k'
        source_interface = dict(waveform_path=expected_path, source_sample_rate=SAMPLE_RATE,
                                source_input_samples=CLIP_SAMPLES)
        if any(key in meta and meta[key] != value for key, value in source_interface.items()):
            raise ValueError(f'{label}: cache waveform path disagrees with online source audio')
        if not required <= set(values) or (not split_path and set(values) != set(caps)):
            raise ValueError(f'{label}: cache must cover all split/normalization IDs; legacy mode requires matching 2000 IDs')
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
            if 'waveform_path' in meta and meta['waveform_path'] != 'sourcewav->32k->24k':
                raise ValueError('Explicit feature cache waveform path must match online sourcewav->32k->24k')
            if len({tuple(value.shape) for value in values.values()}) != 1:
                raise ValueError('Fixed-duration feature cache must have a consistent sequence length')
            features = {sid: values[sid] for sid in ordered}
        else:
            expected = dict(normalized=True, sample_rate=32000, hop_length=800)
            if any(key in meta and meta[key] != value for key, value in expected.items()):
                raise ValueError('Teacher metadata contradicts normalized native H3 latents')
            if 'waveform_path' in meta and meta['waveform_path'] != 'sourcewav->32k':
                raise ValueError('Explicit teacher waveform path must match online sourcewav->32k')
            teacher = {sid: values[sid] for sid in ordered}
        metadata[label] = meta
        metadata['cache_id_counts'][label] = len(values)
    return split, text, mean, std, teacher, features, metadata


def initialize_adapter(model, path, split, backend, allow_train_mismatch=False):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    init_split = checkpoint.get('split')
    validate_split(init_split)
    if checkpoint.get('backend') != backend or checkpoint.get('interface') != INTERFACE:
        raise ValueError('Warmup initialization requires identical backend and native H3 interface')
    if init_split['val'] != split['val']:
        raise ValueError('Warmup initialization requires identical validation IDs and order')
    if checkpoint['split'] != split and not allow_train_mismatch:
        raise ValueError('Warmup initialization requires identical split unless --allow_init_train_mismatch is explicit')
    if set(init_split['train']) & set(split['val']):
        raise ValueError('Warmup training IDs intersect current validation IDs')
    if checkpoint.get('stage') != 'bridge' or not isinstance(checkpoint.get('adapter'), dict):
        raise ValueError('--init loads only a warmup adapter checkpoint, not a full model/resume')
    normalization = checkpoint.get('normalization', checkpoint.get('input_metadata', {}).get('normalization'))
    if normalization is not None:
        validate_ids(normalization.get('ids'), 'init normalization_ids')
        if set(normalization['ids']) & set(split['val']):
            raise ValueError('Warmup normalization IDs intersect current validation IDs')
    model.adapter.load_state_dict(checkpoint['adapter'], strict=True)
    return dict(path=str(path), train_count=len(init_split['train']), val_count=len(init_split['val']),
                train_sha256=ids_fingerprint(init_split['train']), val_sha256=ids_fingerprint(init_split['val']),
                same_split=init_split == split, same_train=init_split['train'] == split['train'],
                current_train_subset_of_init=set(split['train']) <= set(init_split['train']),
                allow_train_mismatch=allow_train_mismatch, validation_train_overlap_count=0,
                normalization=normalization, legacy_normalization_ids_missing=normalization is None)


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
    wall_start_unix = time.time()
    wall_start = time.perf_counter()
    performance = dict(setup_seconds=0.0, input_load_seconds=0.0, train_data_load_seconds=0.0,
                       forward_backward_optimizer_seconds=0.0, validation_seconds=0.0,
                       checkpoint_seconds=0.0, export_seconds=0.0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('bridge', 'clip', 'bridge_clip'))
    parser.add_argument('--backend', required=True, choices=('muq', 'musicfm'))
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--caps', required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--split', default=None, help='JSON object containing explicit ordered train and val ID lists')
    parser.add_argument('--normalization_ids', default=None, help='JSON string list for fixed text normalization statistics, disjoint from val')
    parser.add_argument('--teacher', default=None)
    parser.add_argument('--features', default=None)
    parser.add_argument('--init', default=None, help='Load adapter only from same-split bridge warmup')
    parser.add_argument('--allow_init_train_mismatch', action='store_true', help='Explicitly allow warmup train IDs to differ; val/backend/interface must match')
    parser.add_argument('--out', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--max_steps', type=int, default=0, help='Absolute total optimizer updates; positive value overrides epoch limit')
    parser.add_argument('--eval_every_steps', type=int, default=0, help='Validate/checkpoint every N optimizer updates; zero uses every five epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Per-rank batch size')
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--adapter_lr', type=float, default=1e-4)
    parser.add_argument('--head_lr', type=float, default=1e-4)
    parser.add_argument('--bridge_weight', type=float, default=1.0)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        parser.error('Use a nonexistent output directory; existing directories will not be reused')
    if args.max_steps < 0 or args.eval_every_steps < 0:
        parser.error('max_steps and eval_every_steps must be nonnegative')
    if args.allow_init_train_mismatch and not args.init:
        parser.error('--allow_init_train_mismatch requires --init')
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
    load_start = time.perf_counter()
    split, text, mean, std, teacher, features, metadata = load_inputs(args)
    performance['input_load_seconds'] = time.perf_counter() - load_start
    model = MusicToH3(DenseMusicEncoder(args.backend, args.model_path))
    metadata['backbone'] = model.backbone.metadata()
    initialization = dict(backbone='public initialization', adapter=args.init or 'random', audio_head='random',
                          optimizer='fresh; --init transfers adapter weights only, not optimizer or steps')
    if args.init:
        initialization['adapter_audit'] = initialize_adapter(model, args.init, split, args.backend,
                                                             args.allow_init_train_mismatch)
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
              ('teacher', args.teacher), ('features', args.features), ('split', args.split),
              ('normalization_ids', args.normalization_ids), ('init', args.init)) if path} if rank == 0 else {}
    global_batch = world * args.batch_size
    target_steps = args.max_steps or args.epochs * len(loader)
    global_step, epoch, batch, last_eval_step = 0, 0, 0, 0
    train_wall_seconds, window_seconds, window_steps = 0.0, 0.0, 0
    gpu_events = []
    gradients = {}

    def progress():
        return dict(global_step=global_step, target_steps=target_steps, world_size=world,
                    global_batch=global_batch, epoch=epoch, batch=batch,
                    epoch_completed=batch == len(loader), epochs_completed=global_step // len(loader),
                    actually_items_seen=global_step * global_batch,
                    rank_items_seen=global_step * args.batch_size,
                    steps_per_epoch=len(loader), train_unique_ids=len(split['train']),
                    epoch_discarded_ids=len(split['train']) - len(loader) * global_batch)

    def memory_peaks():
        return dict(cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                    cuda_peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)

    def timing_snapshot():
        wall_seconds = time.perf_counter() - wall_start
        return dict(performance, train_wall_seconds=train_wall_seconds,
                    train_steps_per_second=global_step / max(train_wall_seconds, 1e-12),
                    train_items_per_second=global_step * global_batch / max(train_wall_seconds, 1e-12),
                    wall_start_unix=wall_start_unix, wall_end_unix=time.time(), wall_seconds=wall_seconds,
                    wall_steps_per_second=global_step / max(wall_seconds, 1e-12),
                    wall_items_per_second=global_step * global_batch / max(wall_seconds, 1e-12))

    def write_record(record):
        if rank == 0:
            with (out / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            print(json.dumps(record, allow_nan=False), flush=True)

    def validation_checkpoint(reason, sums, batches):
        nonlocal best_score, best_metrics, last_eval_step
        if global_step < 1:
            raise RuntimeError('Refuse a step-zero training checkpoint')
        if world > 1:
            dist.barrier()
        if rank == 0:
            validation_start = time.perf_counter()
            metrics = evaluate(forward, val_loader, device, baseline.get('val_bridge_MSE'))
            performance['validation_seconds'] += time.perf_counter() - validation_start
            checkpoint_start = time.perf_counter()
            checkpoint = dict(adapter=cpu_state(model.adapter), backend=args.backend, model_path=args.model_path,
                              config=vars(args), split=split, metrics=metrics, baseline=baseline,
                              input_sha256=hashes, input_metadata=metadata, text_mean=mean, text_std=std,
                              normalization=metadata['normalization'], initialization=initialization,
                              stage=args.stage, interface=INTERFACE, gradients=gradients,
                              performance=timing_snapshot(), **progress(), **memory_peaks())
            if args.stage != 'bridge':
                checkpoint['model'] = cpu_state(model)
            atomic_save(checkpoint, out / 'last.pt')
            score = -metrics['val_bridge_MSE'] if args.stage == 'bridge' else metrics['val_R@10']
            if score > best_score:
                best_score, best_metrics = score, metrics
                atomic_save(checkpoint, out / 'best.pt')
            del checkpoint
            performance['checkpoint_seconds'] += time.perf_counter() - checkpoint_start
        if world > 1:
            dist.barrier()
        epoch_sums = sums.clone()
        if world > 1:
            dist.all_reduce(epoch_sums)
        if rank == 0:
            write_record(dict(event='validation', reason=reason, stage=args.stage, **progress(), **metrics,
                              **dict(zip(('train_loss', 'train_clip_loss', 'train_bridge_MSE'),
                                         (epoch_sums / (world * batches)).tolist())),
                              performance=dict(performance), **memory_peaks()))
        last_eval_step = global_step
        train_model.train()
        if args.stage == 'bridge':
            model.backbone.eval()

    performance['setup_seconds'] = time.perf_counter() - wall_start - performance['input_load_seconds']
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
        validation_start = time.perf_counter()
        baseline = evaluate(forward, val_loader, device)
        performance['validation_seconds'] += time.perf_counter() - validation_start
        write_record(dict(event='setup', stage=args.stage, parameters=counts, baseline=baseline, dtype='float32',
                          subset=metadata['subset'], cache_id_counts=metadata['cache_id_counts'],
                          normalization=metadata['normalization'], initialization=initialization,
                          input_sha256=hashes, performance=dict(performance), wall_start_unix=wall_start_unix,
                          bn_caveat='bridge backbone eval; joint/CLIP train BN uses per-rank stereo batch; validation uses running stats, not SyncBN',
                          input_samples=CLIP_SAMPLES, latent_frames=LATENT_FRAMES, **progress()))
    if world > 1:
        dist.barrier()
    while global_step < target_steps:
        epoch += 1
        sampler.set_epoch(epoch - 1)
        train_model.train()
        if args.stage == 'bridge':
            model.backbone.eval()
        sums = torch.zeros(3, device=device)
        window_start = time.perf_counter()
        load_start = window_start
        for batch, (source, target, teacher_batch, frames) in enumerate(loader, 1):
            performance['train_data_load_seconds'] += time.perf_counter() - load_start
            next_step = global_step + 1
            final_step = next_step == target_steps
            step_eval = args.eval_every_steps > 0 and next_step % args.eval_every_steps == 0
            epoch_eval = args.eval_every_steps == 0 and batch == len(loader) and epoch % 5 == 0
            should_eval = step_eval or epoch_eval or final_step
            should_log = next_step == 1 or next_step % 20 == 0 or batch == len(loader) or should_eval
            compute_start = time.perf_counter()
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record()
            optimizer.zero_grad(set_to_none=True)
            embedding, latent = train_model(source.to(device, non_blocking=True), frames.to(device))
            finite_guard([('latent', latent)] + ([('embedding', embedding)] if embedding is not None else []), device, 'forward')
            lclip = clip_loss(embedding, target.to(device)) if args.stage != 'bridge' else latent.new_zeros(())
            lbridge = bridge_terms(latent, teacher_batch.to(device), frames.to(device))[0].mean() if args.stage != 'clip' else latent.new_zeros(())
            loss = lbridge if args.stage == 'bridge' else lclip + args.bridge_weight * lbridge
            finite_guard([('loss', loss)], device, 'loss')
            loss.backward()
            report = gradient_report(model, device, detailed=should_log)
            if report:
                gradients = report
            norm = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
            optimizer.step()
            sums += torch.stack([loss.detach(), lclip.detach(), lbridge.detach()])
            end_event.record()
            gpu_events.append((start_event, end_event))
            performance['forward_backward_optimizer_seconds'] += time.perf_counter() - compute_start
            global_step = next_step
            window_steps += 1
            step_record = dict(event='step', stage=args.stage, **progress())
            if should_log:
                finite_guard(((name, p) for name, p in model.named_parameters() if p.requires_grad), device, 'updated weights')
                # Synchronize only at logging boundaries; report observed windows, never future ETA.
                torch.cuda.synchronize(device)
                window_seconds += time.perf_counter() - window_start
                train_wall_seconds += window_seconds
                gpu_seconds = sum(start.elapsed_time(end) for start, end in gpu_events) / 1000
                performance['train_gpu_seconds'] = performance.get('train_gpu_seconds', 0.0) + gpu_seconds
                step_record.update(loss=loss.item(), clip_loss=lclip.item(), bridge_loss=lbridge.item(),
                                   grad_norm=norm.item(), gradients=gradients, **memory_peaks(),
                                   timing=dict(window_steps=window_steps, window_seconds=window_seconds,
                                               mean_step_seconds=window_seconds / window_steps,
                                               window_steps_per_second=window_steps / max(window_seconds, 1e-12),
                                               window_items_per_second=window_steps * global_batch / max(window_seconds, 1e-12),
                                               window_gpu_seconds=gpu_seconds), performance=dict(performance))
                gpu_events.clear()
                window_seconds, window_steps = 0.0, 0
            write_record(step_record)
            if should_eval:
                validation_checkpoint('final' if final_step else 'step_interval' if step_eval else 'epoch_interval', sums, batch)
            if global_step == target_steps:
                break
            if should_log:
                window_start = time.perf_counter()
            load_start = time.perf_counter()
    if global_step != target_steps or global_step < 1 or last_eval_step != global_step:
        raise RuntimeError('Training did not finish requested updates and final validation/checkpoint')
    if rank == 0:
        export_start = time.perf_counter()
        summary = dict(config=vars(args), split=split, parameters=counts, baseline=baseline,
                       best_metrics=best_metrics, input_sha256=hashes, input_metadata=metadata,
                       normalization=metadata['normalization'], gradients=gradients,
                       checkpoint_sha256={name: sha256(out / (name + '.pt')) for name in ('last', 'best')},
                       initialization=initialization, **progress(), **memory_peaks(),
                       bn_caveat='per-rank BN train stats versus validation running stats; bridge backbone always eval', dtype='float32')
        performance['export_seconds'] += time.perf_counter() - export_start
        summary.update(performance=timing_snapshot())
        with (out / 'summary.json').open('w') as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
        write_record(dict(event='finished', **progress(), performance=summary['performance'], **memory_peaks()))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
