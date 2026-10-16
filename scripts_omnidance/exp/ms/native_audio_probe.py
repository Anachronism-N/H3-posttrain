"""原生H3音频编码侧的接口/梯度检查，不训练DiT，不检索caption。
缓存pre_block输出；验证经mean_proj/固定归一化后与encode_audio完全同形。
--text提供冻结H3文本向量时，做一次CLIP反向检查，仅音频mean_proj和音频侧投影有梯度。
这是接口测试，不是音乐控制效果或已完成训练的证明。
"""
import argparse
import hashlib
import json
import os
import sys

import torch
import torch.nn.functional as F

ROOT = '/apdcephfs_gy4/share_302533218/cedricnie'
sys.path.insert(0, ROOT + '/DiffSynth-Studio-upstream-20260930')


def main():
    from safetensors.torch import load_file
    import soundfile as sf
    import torchaudio
    from diffsynth.models.minimax_h3_audio_vae import MiniMaxH3AudioVAE

    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--ids')
    source.add_argument('--caps', help='Selected caption JSON; only its clip IDs are used for native feature caching')
    ap.add_argument('--out', required=True)
    ap.add_argument('--text', default='')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--verify_n', type=int, default=3)
    ap.add_argument('--w', default='/dev/shm/h3/Ref2VA/audio_vae/model.safetensors')
    a = ap.parse_args()
    ids = sorted(json.load(open(a.caps))) if a.caps else [line.strip() for line in open(a.ids) if line.strip()]
    if not 0 <= a.shard < a.nshard:
        raise ValueError('Invalid native cache shard')
    ids = ids[a.shard::a.nshard]
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    model = MiniMaxH3AudioVAE()
    state = load_file(a.w)
    for name in ('q_bias', 'zero_k_bias', 'v_bias'):
        key = f'pre_block.attn.{name}'
        state[key + '.weight'] = state.pop(key)  # DiffSynth wraps these checkpoint tensors in WarpedTensor.
    model.load_state_dict(state, strict=True)
    del state
    model = model.to('cuda', torch.float32).eval().requires_grad_(False)
    mean, std = model._latent_stats('cuda')
    results, cached = {}, {}
    with torch.no_grad():
        for index, sid in enumerate(ids):
            path = ROOT + '/OmniDance_processed/pose_preprocess_outputs/mainline/normalized_704x1280/main/audio/' + sid + '.wav'
            audio, sr = sf.read(path, dtype='float32', always_2d=True)
            waveform = torch.from_numpy(audio.T.copy()).cuda()
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            if waveform.shape[0] != 2:
                raise ValueError(f'Expected mono or stereo audio: {path}')
            waveform = torchaudio.functional.resample(waveform, sr, model.sample_rate)
            padded = model.preprocess(waveform[:, None])
            z = model.encoder(padded)
            z = model.pre_block(z.transpose(1, 2)).transpose(1, 2)
            latent = (model.mean_proj(z) - mean) / std
            error = None
            if index < a.verify_n:
                reference = model.encode_audio(waveform)
                assert latent.shape == reference.shape
                error = (latent - reference).abs().max().item()
                assert error < 1e-4
            assert torch.isfinite(latent).all()
            cached[sid] = z.cpu()
            results[sid] = dict(shape=list(latent.shape), reference_max_abs_error=error,
                                effective_hz=model.sample_rate / model.hop_length,
                                latent_rms=latent.square().mean().sqrt().item())
            if (index + 1) % 25 == 0:
                print('native cache', index + 1, '/', len(ids), flush=True)
    torch.save(dict(features=cached, mean_proj={k: v.cpu() for k, v in model.mean_proj.state_dict().items()},
                    mean=mean.cpu(), std=std.cpu(), sample_rate=model.sample_rate,
                    hop_length=model.hop_length), os.path.join(a.out, 'native_features.pt'))
    report = dict(interface=results, clip_gradient_test=None, status='interface_only')
    if a.text:
        targets = torch.load(a.text, map_location='cpu', weights_only=False)
        batch = [sid for sid in ids if sid in targets]
        if len(batch) < 2:
            raise ValueError('CLIP gradient check needs at least two paired items')
        lengths = {cached[sid].shape for sid in batch}
        if len(lengths) != 1:
            raise ValueError('Use equal-length clips for this small gradient check')
        z = torch.stack([cached[sid] for sid in batch]).cuda()
        model.mean_proj.requires_grad_(True)
        lat = model.mean_proj(z.flatten(0, 1)).reshape(len(batch), 2, 32, -1)
        lat = (lat - mean) / std
        pooled = torch.cat([lat.mean(-1).flatten(1), lat.std(-1).flatten(1)], 1)
        target = torch.stack([targets[sid].float() for sid in batch]).cuda()
        head = torch.nn.Linear(pooled.shape[1], target.shape[1]).cuda()
        prediction = F.normalize(head(pooled), dim=-1)
        logits = prediction @ F.normalize(target, dim=-1).T / 0.07
        labels = torch.arange(len(batch), device='cuda')
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        loss.backward()
        grads = {name: parameter.grad.norm().item() for name, parameter in model.mean_proj.named_parameters()}
        assert torch.isfinite(loss) and all(torch.isfinite(parameter.grad).all() for parameter in model.mean_proj.parameters())
        assert grads['weight'] > 0 and all(parameter.grad is None for parameter in model.encoder.parameters())
        report.update(status='interface_and_gradient_only', clip_gradient_test=dict(loss=loss.item(), mean_proj_grad_norm=grads,
                      training_parameters='audio mean_proj and audio-side projection only', text_trainable=False, dit_loaded=False))
    with open(a.w, 'rb') as stream:
        report['weights_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    with open(os.path.join(a.out, 'report.json'), 'w') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
