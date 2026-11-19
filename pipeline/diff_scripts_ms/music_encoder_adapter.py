"""Shared dense music backbone and native H3 latent adapter.
24kHz input, separate stereo channels; no caption retrieval or native-encoder bypass.
Frozen feature caches are for probes/bridge warmup only, not full encoder finetuning.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
AUDIO_ROOT = ROOT / 'OmniDance_processed/pose_preprocess_outputs/mainline/normalized_704x1280/main/audio'
DEFAULT_PATHS = {
    'muq': '/opt/models/MuQ-large-msd-iter',
    'musicfm': '/opt/models/MusicFM-MSD',
    'mert': '/opt/models/MERT-v1-330M',
}


class DenseMusicEncoder(nn.Module):
    sample_rate = 24000
    hidden_size = 1024

    def __init__(self, backend, model_path=None, musicfm_source='/opt/musicfm'):
        super().__init__()
        self.backend = backend
        self.model_path = model_path or DEFAULT_PATHS[backend]
        self.frame_hz = 75.0 if backend == 'mert' else 25.0
        self.frame_offset = 0.0
        if backend == 'muq':
            from muq import MuQ
            self.model = MuQ.from_pretrained(self.model_path)
            if self.model.config.label_rate != 25 or self.model.config.encoder_dim != 1024:
                raise ValueError('Unexpected MuQ frame rate or feature width')
            # SSL target heads/quantizers are unused by dense feature extraction.
            self.model.model.rvq.requires_grad_(False)
            self.model.model.linear.requires_grad_(False)
            self.model.model.cls_token.requires_grad_(False)
        elif backend == 'musicfm':
            sys.path.insert(0, str(Path(musicfm_source).parent))
            from musicfm.model.musicfm_25hz import MusicFM25Hz
            self.model = MusicFM25Hz(is_flash=False, stat_path=str(Path(self.model_path) / 'msd_stats.json'),
                                    model_path=str(Path(self.model_path) / 'pretrained_msd.pt'))
            self.model.linear.requires_grad_(False)
            self.model.cls_token.requires_grad_(False)
            for name, module in self.model.named_children():
                if name.startswith('quantizer_'):
                    module.requires_grad_(False)
        elif backend == 'mert':
            from transformers import AutoModel
            # This local official source was inspected before trust_remote_code.
            self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)
            self.model.config.apply_spec_augment = False
            stride, receptive_field = 1, 1
            for kernel, factor in zip(self.model.config.conv_kernel, self.model.config.conv_stride):
                receptive_field += (kernel - 1) * stride
                stride *= factor
            self.frame_hz = self.sample_rate / stride
            self.frame_offset = (receptive_field - 1) / (2 * self.sample_rate)
        else:
            raise ValueError(backend)
        self.model.float()

    def forward(self, waveform):
        """[N,L] raw mono channels -> [N,T,1024], gradients to actual backbone."""
        if waveform.ndim != 2 or not torch.isfinite(waveform).all():
            raise ValueError('Expected finite [N,L] waveform at 24kHz')
        waveform = waveform.float()
        if self.backend == 'mert':
            # Official local preprocessor_config specifies do_normalize=false.
            hidden = self.model(waveform, output_hidden_states=False).last_hidden_state
        else:
            model = self.model.model if self.backend == 'muq' else self.model
            frontend = model.preprocessor_melspec_2048
            frontend.to(waveform.device)
            mel = frontend(waveform)[..., :-1]
            mel = (mel - model.stat['melspec_2048_mean']) / model.stat['melspec_2048_std']
            conv = model.conv(mel)
            # Bypass SSL prediction head and no_grad convenience wrappers.
            hidden = model.conformer(conv, output_hidden_states=False).last_hidden_state
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size or not torch.isfinite(hidden).all():
            raise RuntimeError('Invalid music encoder sequence')
        return hidden.float()

    def metadata(self):
        return dict(backend=self.backend, model_path=self.model_path, sample_rate=self.sample_rate,
                    frame_hz=self.frame_hz, frame_offset_seconds=self.frame_offset,
                    hidden_size=self.hidden_size, parameters=sum(p.numel() for p in self.parameters()),
                    dense_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    scope='frame sequence, no temporal averaging or interpolation in feature cache')


class MusicEncodingProxy:
    """Run the same trained backbone/adapter in its isolated dependency environment.

    H3's text encoder and music backbones use different Transformers versions.
    Only exact-waveform results are memoized; this is not caption retrieval.
    """
    def __init__(self, checkpoint, python='/opt/venv_music/bin/python'):
        self.checkpoint = str(Path(checkpoint).resolve())
        self.python = python
        self.cached_hash = None
        self.cached_latent = None
        self.calls = 0

    def __call__(self, waveform, dtype=None):
        import subprocess
        import tempfile
        waveform_cpu = waveform.detach().cpu().float().contiguous()
        if waveform_cpu.ndim != 2 or waveform_cpu.shape[0] != 2 or not torch.isfinite(waveform_cpu).all():
            raise ValueError('Music replacement expects finite stereo [2,L] at 32kHz')
        fingerprint = hashlib.sha256(waveform_cpu.numpy().tobytes()).hexdigest()
        if fingerprint != self.cached_hash:
            with tempfile.TemporaryDirectory(prefix='h3_music_encode_') as directory:
                input_path, output_path = Path(directory) / 'waveform.pt', Path(directory) / 'latents.pt'
                torch.save(waveform_cpu, input_path)
                environment = os.environ.copy()
                environment.update(OMP_NUM_THREADS='2', HF_HUB_OFFLINE='1')
                result = subprocess.run([self.python, str(Path(__file__).resolve()),
                                         '--checkpoint', self.checkpoint, '--encode_input', str(input_path),
                                         '--out', str(output_path)], env=environment, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError('Music encoder subprocess failed: ' + result.stderr[-6000:])
                output = torch.load(output_path, map_location='cpu', weights_only=True)
                expected = (2, 32, math.ceil(waveform_cpu.shape[-1] / 800))
                if output.shape != expected or not torch.isfinite(output).all():
                    raise ValueError('Music replacement output has invalid shape or values')
                self.cached_latent = output
                self.cached_hash = fingerprint
        self.calls += 1
        print(f'[music replacement] encode call={self.calls} waveform={fingerprint[:12]} shape={list(self.cached_latent.shape)}', flush=True)
        return self.cached_latent.to(device=waveform.device, dtype=waveform.dtype)


def load_music_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    backend = checkpoint['backend']
    backbone = DenseMusicEncoder(backend, checkpoint['model_path'])
    model = MusicToH3(backbone)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
    elif checkpoint.get('stage') == 'bridge':
        model.adapter.load_state_dict(checkpoint['adapter'], strict=True)
    else:
        raise ValueError('Unsupported music replacement checkpoint')
    model = model.float().eval()
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError('Non-finite trained music encoder weights')
    return model


class TemporalLatentAdapter(nn.Module):
    def __init__(self, feature_dim=1024, width=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, width), nn.GELU())
        self.local = nn.Conv1d(width, width, 3, padding=1)
        self.output = nn.Linear(width, 32)

    def forward(self, hidden, target_frames, source_hz, source_offset=0.0):
        x = self.net(hidden)
        x = x + F.gelu(self.local(x.transpose(1, 2)).transpose(1, 2))
        # Positions are explicit elapsed seconds, not normalized endpoints.
        position = (torch.arange(target_frames, device=x.device, dtype=torch.float32) / 40.0 - source_offset) * source_hz
        position = position.clamp(0, x.shape[1] - 1)
        low = position.floor().long()
        high = (low + 1).clamp(max=x.shape[1] - 1)
        weight = (position - low).reshape(1, -1, 1)
        aligned = x[:, low] * (1 - weight) + x[:, high] * weight
        return self.output(aligned).transpose(1, 2).contiguous()


class MusicToH3(nn.Module):
    """Shared normalized 32-d generator latent; CLIP reads this same latent."""
    sample_rate = 32000
    hop_length = 800

    def __init__(self, backbone, adapter=None):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter or TemporalLatentAdapter(backbone.hidden_size)
        self.audio_head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 5120, bias=False))

    def encode_stereo(self, waveform):
        import torchaudio
        if waveform.ndim != 3 or waveform.shape[1] != 2:
            raise ValueError('Expected [B,2,L] at 32kHz')
        batch, _, length = waveform.shape
        frames = math.ceil(length / self.hop_length)
        mono = waveform.reshape(batch * 2, length)
        mono = torchaudio.functional.resample(mono, self.sample_rate, self.backbone.sample_rate)
        hidden = self.backbone(mono)
        normalized = self.adapter(hidden, frames, self.backbone.frame_hz, self.backbone.frame_offset)
        return normalized.reshape(batch, 2, 32, frames)

    def forward(self, waveform, return_latent=False, effective_frames=None):
        latent = self.encode_stereo(waveform)
        if effective_frames is None:
            effective_frames = torch.full((len(latent),), latent.shape[-1], device=latent.device)
        if effective_frames.shape != (len(latent),) or torch.any((effective_frames < 1) | (effective_frames > latent.shape[-1])):
            raise ValueError('Invalid effective frame mask')
        mask = (torch.arange(latent.shape[-1], device=latent.device)[None, :] < effective_frames[:, None])[:, None, None]
        denominator = effective_frames.float()[:, None, None]
        mean = (latent * mask).sum(-1) / denominator
        variance = ((latent - mean[..., None]).square() * mask).sum(-1) / denominator
        pooled = torch.cat([mean.flatten(1), variance.clamp_min(1e-12).sqrt().flatten(1)], 1)
        embedding = F.normalize(self.audio_head(pooled).float(), dim=-1)
        return (embedding, latent) if return_latent else embedding

    @torch.no_grad()
    def encode_audio(self, waveform, dtype=None):
        # Pipeline calls this for reference audio and for retake target audio.
        if waveform.ndim != 2:
            raise ValueError('Expected [2,L] pipeline input')
        latent = self.encode_stereo(waveform.float().unsqueeze(0))[0]
        if not torch.isfinite(latent).all():
            raise RuntimeError('Non-finite H3 adapter output')
        return latent.to(dtype=waveform.dtype)


def load_waveform(sid):
    import soundfile as sf
    import torchaudio
    with sf.SoundFile(str(AUDIO_ROOT / (sid + '.wav'))) as stream:
        sr = stream.samplerate
        audio = stream.read(math.ceil(5.0625 * sr), dtype='float32', always_2d=True)
    waveform = torch.from_numpy(audio.T.copy())
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    if waveform.shape[0] != 2 or waveform.shape[-1] == 0 or not torch.isfinite(waveform).all():
        raise ValueError(f'Invalid source waveform: {sid}')
    waveform = torchaudio.functional.resample(waveform, sr, 24000)
    samples = 121500
    return F.pad(waveform[:, :samples], (0, max(0, samples - waveform.shape[-1])))


def cache_features(args):
    ids = sorted(json.load(open(args.caps)))[args.shard::args.nshard]
    backbone = DenseMusicEncoder(args.backend, args.model_path).cuda().eval()
    features = {}
    with torch.no_grad():
        for index in range(0, len(ids), args.batch_size):
            chunk = ids[index:index + args.batch_size]
            wav = torch.stack([load_waveform(sid) for sid in chunk]).cuda()
            sequence = backbone(wav.flatten(0, 1)).reshape(len(chunk), 2, -1, backbone.hidden_size)
            for sid, value in zip(chunk, sequence.cpu()):
                features[sid] = value.half()
            print(args.backend, index + len(chunk), '/', len(ids), 'shape', list(sequence.shape), flush=True)
    result = dict(features=features, metadata=backbone.metadata())
    path = Path(args.out)
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    torch.save(result, str(path) + '.tmp')
    os.replace(str(path) + '.tmp', path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    print('DONE', len(features), digest, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=list(DEFAULT_PATHS))
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--caps')
    parser.add_argument('--checkpoint')
    parser.add_argument('--encode_input', help='Stereo 32kHz waveform tensor for isolated H3 inference')
    parser.add_argument('--out', required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--nshard', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()
    if args.encode_input:
        if not args.checkpoint:
            parser.error('--encode_input requires --checkpoint')
        waveform = torch.load(args.encode_input, map_location='cpu', weights_only=True)
        model = load_music_checkpoint(args.checkpoint).cuda()
        output = model.encode_audio(waveform.cuda()).cpu()
        torch.save(output, args.out)
        print('ENCODED', list(output.shape), flush=True)
    else:
        if not args.backend or not args.caps:
            parser.error('Feature cache extraction requires --backend and --caps')
        if not 0 <= args.shard < args.nshard:
            parser.error('Invalid shard')
        cache_features(args)
