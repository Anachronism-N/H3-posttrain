"""H3原生audio-VAE完整编码侧CLIP；冻结文本，只训练encoder/pre_block/mean_proj与音频head。
单卡或torchrun --standalone --nproc_per_node=8；无DiT、文本hook、重建或视频loss。
验证检索仅为同源heldout表征诊断；推理仅加载native_audio_encoder，不加载5120维head。
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import types

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.nn.functional import all_gather as autograd_all_gather
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.data import DataLoader, Dataset, DistributedSampler

ROOT = '/apdcephfs_gy4/share_302533218/cedricnie'
AUDIO_ROOT = ROOT + '/OmniDance_processed/pose_preprocess_outputs/mainline/normalized_704x1280/main/audio'
SAMPLE_RATE, CLIP_SAMPLES = 32000, 162000  # 前81/16秒；hop=800时统一补到162400。


class AudioPairs(Dataset):
    def __init__(self, ids, targets, hop_length, audio_root=AUDIO_ROOT):
        self.ids, self.targets, self.audio_root = ids, targets, audio_root
        self.padded_samples = math.ceil(CLIP_SAMPLES / hop_length) * hop_length

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        import soundfile as sf
        import torchaudio
        sid = self.ids[index]
        path = os.path.join(self.audio_root, sid + '.wav')
        with sf.SoundFile(path) as stream:
            sr = stream.samplerate
            audio = stream.read(math.ceil(CLIP_SAMPLES * sr / SAMPLE_RATE), dtype='float32', always_2d=True)
        if audio.ndim != 2 or audio.shape[1] not in (1, 2) or not len(audio) or sr <= 0:
            raise ValueError(f'Expected nonempty original mono or stereo audio: {path}')
        waveform = torch.from_numpy(audio.T.copy())
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)  # Same mono-to-stereo conversion as native H3 inference.
        if not torch.isfinite(waveform).all():
            raise ValueError(f'Non-finite source audio: {path}')
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        waveform = waveform[:, :CLIP_SAMPLES]
        waveform = F.pad(waveform, (0, self.padded_samples - waveform.shape[-1]))
        if waveform.shape != (2, self.padded_samples) or not torch.isfinite(waveform).all():
            raise ValueError(f'Invalid resampled audio: {path}')
        return waveform, self.targets[sid]


def checkpoint_encoder_forward(encoder, waveform):
    return checkpoint_sequential(encoder.block, len(encoder.block), waveform, use_reentrant=False)


class NativeAudioEncoder(nn.Module):
    def __init__(self, native, checkpoint_encoder=False):
        super().__init__()
        native.requires_grad_(False)
        for name in ('encoder', 'pre_block', 'mean_proj'):
            getattr(native, name).requires_grad_(True)
        # 未用解码侧不注册进DDP，也不搬到GPU；推理仍使用原始完整VAE。
        for name in ('decoder', 'dec_in_proj', 'logs_proj'):
            delattr(native, name)
        if checkpoint_encoder:
            native.encoder.forward = types.MethodType(checkpoint_encoder_forward, native.encoder)
        self.native = native
        mean, std = native._latent_stats('cpu')
        self.register_buffer('mean', mean.float().reshape(1, 1, 32, 1))
        self.register_buffer('std', std.float().reshape(1, 1, 32, 1))
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 512), nn.GELU(), nn.Linear(512, 5120))

    def latent(self, waveform):
        batch = waveform.shape[0]
        raw = self.native.encode(self.native.preprocess(waveform.reshape(batch * 2, 1, -1)))
        if raw.ndim != 3 or raw.shape[:2] != (batch * 2, 32):
            raise RuntimeError(f'Unexpected native latent shape: {tuple(raw.shape)}')
        return (raw.float().reshape(batch, 2, 32, -1) - self.mean) / self.std

    def forward(self, waveform):
        latent = self.latent(waveform)
        pooled = torch.cat([latent.mean(-1).flatten(1), latent.std(-1).flatten(1)], dim=1)
        return F.normalize(self.head(pooled).float(), dim=-1)


def clip_loss(audio, text):
    if dist.is_initialized():
        # 每rank计算同一个全局loss；gather反传求和，DDP参数梯度取平均，等价全局单卡loss。
        audio = torch.cat(autograd_all_gather(audio.contiguous()))
        gathered = [torch.empty_like(text) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, text.contiguous())
        text = torch.cat(gathered)
    scores = audio.float() @ text.float().T / 0.07
    labels = torch.arange(len(audio), device=audio.device)
    return (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)) / 2


def finite_guard(named_tensors, device, stage, synchronize=True):
    pairs = list(named_tensors)
    valid = torch.stack([torch.isfinite(t).all() if t is not None else torch.tensor(False, device=device)
                         for _, t in pairs]).all().to(dtype=torch.int32, device=device)
    if synchronize and dist.is_initialized():
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if not valid.item():
        bad = [name for name, t in pairs if t is None or not torch.isfinite(t).all().item()]
        raise RuntimeError(f'{stage}: non-finite/missing tensors on this rank: {bad}; a peer may have failed')


def cpu_state(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def evaluate(model, loader, device):
    model.eval()
    audio, text = [], []
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for waveform, target in loader:
            audio.append(model(waveform.to(device, non_blocking=True)))
            text.append(target.to(device, non_blocking=True))
    audio, text = torch.cat(audio), torch.cat(text)
    finite_guard([('audio', audio), ('text', text)], device, 'validation', synchronize=False)
    scores = audio @ text.T
    correct = torch.arange(len(audio), device=device)[:, None]
    metrics = {}
    for direction, matrix in (('a2t', scores), ('t2a', scores.T)):
        ranking = matrix.argsort(dim=1, descending=True)
        for k in (1, 5, 10):
            metrics[f'{direction}_R@{k}'] = (ranking[:, :k] == correct).any(1).float().mean().item()
    return metrics


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--caps', required=True, help='2000条selected JSON：舞种+原版MotionScript目标')
    parser.add_argument('--text', required=True, help='冻结的{id: Tensor[5120]}缓存，不训练文本侧')
    parser.add_argument('--w', default='/dev/shm/h3/Ref2VA/audio_vae/model.safetensors')
    parser.add_argument('--out', required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=2, help='每GPU batch size')
    parser.add_argument('--native_lr', type=float, default=1e-6)
    parser.add_argument('--head_lr', type=float, default=3e-4)
    parser.add_argument('--checkpoint_encoder', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.native_lr <= 0 or args.head_lr <= 0:
        raise ValueError('Positive epochs, batch_size and learning rates are required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(42)
    random.seed(42)
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if args.batch_size * world < 2:
        raise ValueError('InfoNCE requires at least two global paired items')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl', device_id=device)
    sys.path.insert(0, ROOT + '/DiffSynth-Studio-upstream-20260930')
    from diffsynth.models.minimax_h3_audio_vae import MiniMaxH3AudioVAE
    from safetensors.torch import load_file
    with open(args.caps) as stream:
        caps = json.load(stream)
    targets = torch.load(args.text, map_location='cpu', weights_only=False)
    if not isinstance(caps, dict) or not isinstance(targets, dict) or len(caps) != 2000 or set(caps) != set(targets):
        raise ValueError('Exactly 2000 selected IDs and matching frozen text vectors are required')
    if any(not isinstance(sid, str) or '/' in sid or '\\' in sid or sid in ('.', '..') for sid in caps):
        raise ValueError('Invalid selected clip ID')
    ids = sorted(caps)
    random.Random(42).shuffle(ids)
    val_ids, train_ids = ids[:200], ids[200:]
    text = torch.stack([targets[sid].detach().float() for sid in train_ids + val_ids])
    if text.shape != (2000, 5120) or not torch.isfinite(text).all():
        raise ValueError('Text cache must contain finite 5120-dimensional vectors')
    text_mean, text_std = text[:1800].mean(0), text[:1800].std(0).clamp_min(1e-4)
    text = F.normalize((text - text_mean) / text_std, dim=-1)
    targets = dict(zip(train_ids + val_ids, text))
    native = MiniMaxH3AudioVAE().float()
    state = load_file(args.w)
    for name in ('q_bias', 'zero_k_bias', 'v_bias'):
        key = f'pre_block.attn.{name}'
        state[key + '.weight'] = state.pop(key)
    native.load_state_dict(state, strict=True)
    del state
    if native.sample_rate != SAMPLE_RATE or native.hop_length != 800:
        raise ValueError('Expected native H3 32kHz/40Hz audio interface')
    model = NativeAudioEncoder(native, args.checkpoint_encoder).to(device, torch.float32)
    finite_guard(model.named_parameters(), device, 'initial weights')
    train_data = AudioPairs(train_ids, targets, native.hop_length)
    sampler = DistributedSampler(train_data, num_replicas=world, rank=rank, seed=42, drop_last=True)
    loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler, drop_last=True,
                        num_workers=2, pin_memory=True, persistent_workers=True)
    if not len(loader):
        raise ValueError('Training split is smaller than the global batch')
    val_loader = DataLoader(AudioPairs(val_ids, targets, native.hop_length), batch_size=args.batch_size,
                            num_workers=2, pin_memory=True) if rank == 0 else None
    train_model = DDP(model, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else model
    optimizer = torch.optim.AdamW([
        {'params': model.native.parameters(), 'lr': args.native_lr, 'weight_decay': 0.0},
        {'params': model.head.parameters(), 'lr': args.head_lr}], weight_decay=0.01)
    split = dict(train=train_ids, val=val_ids, seed=42)
    log, best_score, best_metrics = None, -1.0, None
    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        log = open(os.path.join(args.out, 'metrics.jsonl'), 'a')
        with open(os.path.join(args.out, 'split.json'), 'w') as stream:
            json.dump(split, stream)
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch - 1)
        train_model.train()
        loss_sum = torch.zeros((), device=device)
        for batch_index, (waveform, target) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                embedding = train_model(waveform.to(device, non_blocking=True))
            loss = clip_loss(embedding, target.to(device, non_blocking=True))
            finite_guard([('loss', loss)], device, 'CLIP loss')
            loss.backward()
            finite_guard(((name, p.grad) for name, p in model.named_parameters()), device, 'gradients')
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            finite_guard(model.named_parameters(), device, 'updated weights')
            loss_sum += loss.detach()
            if rank == 0 and (batch_index == 0 or (batch_index + 1) % 20 == 0):
                print(json.dumps(dict(epoch=epoch, batch=batch_index + 1, batches=len(loader),
                                      loss=loss.item(), cuda_peak_gib=torch.cuda.max_memory_allocated(device) / 2**30)), flush=True)
        record = dict(epoch=epoch, train_loss=(loss_sum / len(loader)).item(), stage='representation_only')
        if epoch % 5 == 0 or epoch == args.epochs:
            # 全rank等待rank0原模型验证/CPU保存，避免DDP前向触发未匹配collective。
            if world > 1:
                dist.barrier()
            if rank == 0:
                metrics = evaluate(model, val_loader, device)
                record.update(metrics)
                checkpoint = dict(native_audio_encoder=cpu_state(model.native), head=cpu_state(model.head),
                                  text_mean=text_mean, text_std=text_std, args=vars(args), split=split,
                                  epoch=epoch, metrics=metrics, stage='representation_only')
                torch.save(checkpoint, os.path.join(args.out, 'last.pt'))
                score = (metrics['a2t_R@10'] + metrics['t2a_R@10']) / 2
                if score > best_score:
                    best_score, best_metrics = score, metrics
                    torch.save(checkpoint, os.path.join(args.out, 'best.pt'))
            if world > 1:
                dist.barrier()
        if rank == 0:
            log.write(json.dumps(record, allow_nan=False) + '\n')
            log.flush()
            print(json.dumps(record), flush=True)
    if rank == 0:
        log.close()
        summary = dict(best_metrics=best_metrics, args=vars(args), stage='representation_only',
                       sha256={name: sha256(path) for name, path in (
                           ('base_weights', args.w), ('text', args.text), ('caps', args.caps),
                           ('best', os.path.join(args.out, 'best.pt')), ('last', os.path.join(args.out, 'last.pt')))},
                       trainable_parameters={name: sum(p.numel() for p in module.parameters()) for name, module in (
                           ('encoder', model.native.encoder), ('pre_block', model.native.pre_block),
                           ('mean_proj', model.native.mean_proj), ('audio_head', model.head))})
        with open(os.path.join(args.out, 'summary.json'), 'w') as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
        print('DONE representation_only', flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
