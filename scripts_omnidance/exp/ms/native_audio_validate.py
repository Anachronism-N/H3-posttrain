"""固定配对音频的编码latent与声学重建验收（不是生成舞蹈质量评分）。
对比原生audio-VAE和音频侧CLIP检查点；不检索caption、不加载DiT。
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

ROOT = '/apdcephfs_gy4/share_302533218/cedricnie'
sys.path.insert(0, ROOT + '/DiffSynth-Studio-upstream-20260930')


def spectral_distance(predicted, target):
    n = min(predicted.shape[-1], target.shape[-1])
    window = torch.hann_window(1024, device=predicted.device)
    p = torch.stft(predicted[..., :n], 1024, 256, window=window, return_complex=True).abs()
    t = torch.stft(target[..., :n], 1024, 256, window=window, return_complex=True).abs()
    return (p.sub(t).norm() / t.norm().clamp_min(1e-8)).item()


def main():
    import soundfile as sf
    import torchaudio
    from safetensors.torch import load_file
    from diffsynth.models.minimax_h3_audio_vae import MiniMaxH3AudioVAE

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ids', required=True)
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--dedicated_music', action='store_true', help='Use each dedicated encoder+adapter through the shared isolation proxy')
    ap.add_argument('--out', required=True)
    ap.add_argument('--w', default='/dev/shm/h3/Ref2VA/audio_vae/model.safetensors')
    a = ap.parse_args()
    ids = [line.strip() for line in open(a.ids) if line.strip()]
    state = load_file(a.w)
    for name in ('q_bias', 'zero_k_bias', 'v_bias'):
        key = f'pre_block.attn.{name}'
        state[key + '.weight'] = state.pop(key)
    model = MiniMaxH3AudioVAE().eval().requires_grad_(False)
    model.load_state_dict(state, strict=True)
    model = model.float().cuda()
    originals = {key: value.clone() for key, value in state.items() if key.startswith(('encoder.', 'pre_block.', 'mean_proj.'))}
    del state
    means, stds = model._latent_stats('cuda')
    data, reference = {}, {}
    with torch.no_grad():
        for sid in ids:
            waveform, sr = sf.read(ROOT + '/OmniDance_processed/pose_preprocess_outputs/mainline/normalized_704x1280/main/audio/' + sid + '.wav', dtype='float32', always_2d=True)
            waveform = torch.from_numpy(waveform.T.copy()).cuda()
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            waveform = torchaudio.functional.resample(waveform, sr, model.sample_rate)
            data[sid] = waveform
            latent = model.encode_audio(waveform)
            reconstructed = model.decode_audio(latent).squeeze(0)
            reference[sid] = (latent, reconstructed)
        results = {}
        for path in a.ckpts:
            model.load_state_dict(originals, strict=False)
            if a.dedicated_music:
                from music_encoder_adapter import MusicEncodingProxy
                encode_audio = MusicEncodingProxy(path)
            else:
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                if 'native_audio_encoder' in checkpoint:
                    native = checkpoint['native_audio_encoder']
                else:
                    native = {'mean_proj.' + key: value for key, value in checkpoint['native_mean_proj'].items()}
                expected = model.state_dict()
                if any(key not in originals or value.shape != expected[key].shape or not torch.isfinite(value).all() for key, value in native.items()):
                    raise ValueError('Invalid native encoder checkpoint')
                model.load_state_dict(native, strict=False)
                encode_audio = model.encode_audio
            rows = {}
            for sid, waveform in data.items():
                latent = encode_audio(waveform)
                baseline, baseline_audio = reference[sid]
                assert latent.shape == baseline.shape and torch.isfinite(latent).all()
                decoded = model.decode_audio(latent).squeeze(0)
                assert torch.isfinite(decoded).all()
                n = min(decoded.shape[-1], waveform.shape[-1], baseline_audio.shape[-1])
                gt = waveform[..., :n]
                current = decoded[..., :n]
                previous = baseline_audio[..., :n]
                rows[sid] = dict(latent_relative_rms=(latent.sub(baseline).square().mean().sqrt() / baseline.square().mean().sqrt()).item(),
                                 output_vs_base_relative_rms=(current.sub(previous).square().mean().sqrt() / previous.square().mean().sqrt().clamp_min(1e-8)).item(),
                                 baseline_waveform_l1=F.l1_loss(previous, gt).item(), waveform_l1=F.l1_loss(current, gt).item(),
                                 baseline_spectral_distance=spectral_distance(previous, gt), spectral_distance=spectral_distance(current, gt))
            results[path] = rows
    parent = os.path.dirname(os.path.abspath(a.out))
    if not os.path.isdir(parent):
        raise FileNotFoundError(parent)
    with open(a.out, 'w') as stream:
        json.dump(dict(metrics=results, scope='audio representation/reconstruction only; not dance generation acceptance'), stream, indent=2, allow_nan=False)
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
