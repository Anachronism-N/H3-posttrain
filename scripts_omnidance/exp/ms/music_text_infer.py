"""Isolated music environment executes the entire trained text adapter, never GT caption."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import torch

from music_text_adapter import MusicToH3Text, load_music_text_checkpoint


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class MusicTextEncodingProxy:
    def __init__(self, checkpoint, python='/opt/venv_music/bin/python', legacy_clip=False):
        self.legacy_clip = bool(legacy_clip)
        self.checkpoint = str(Path(checkpoint).resolve(strict=True))
        self.checkpoint_sha256 = sha256(self.checkpoint)
        self.python = python
        self.cache_key = None
        self.cache = None
        self.calls = 0

    def __call__(self, waveform, context):
        waveform = waveform.detach().cpu().float().contiguous()
        context = context.detach().cpu().float().contiguous()
        if (waveform.shape != (2, 162000) or context.shape != (5120,)
                or not torch.isfinite(waveform).all() or not torch.isfinite(context).all()):
            raise ValueError('Proxy requires shared-loader stereo32k 162000 samples and raw5120 structural context')
        key = hashlib.sha256(self.checkpoint_sha256.encode() + waveform.numpy().tobytes() + context.numpy().tobytes()).hexdigest()
        if key != self.cache_key:
            with tempfile.TemporaryDirectory(prefix='h3_music_text_') as directory:
                input_path, output_path = Path(directory) / 'input.pt', Path(directory) / 'output.pt'
                torch.save(dict(waveform=waveform, context=context), input_path)
                environment = os.environ.copy()
                environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2')
                command = [self.python, str(Path(__file__).resolve()),
                    '--checkpoint', self.checkpoint, '--checkpoint_sha256', self.checkpoint_sha256,
                    '--input', str(input_path), '--out', str(output_path)]
                if self.legacy_clip:
                    command += ['--legacy_clip']
                completed = subprocess.run(command, env=environment, capture_output=True, text=True)
                if completed.returncode:
                    raise RuntimeError('Music text inference failed: ' + completed.stderr[-8000:] + completed.stdout[-4000:])
                data = torch.load(output_path, map_location='cpu', weights_only=True)
                tail = data['tail']
                if (data['checkpoint_sha256'] != self.checkpoint_sha256 or tail.ndim != 2
                        or tail.shape[-1] != 5120 or not 1 <= len(tail) <= data['query_max']
                        or data['predicted_length'] != len(tail) or not torch.isfinite(tail).all()):
                    raise ValueError('Invalid deployed raw text output')
                self.cache = data
                self.cache_key = key
        self.calls += 1
        print('[music text deployment] ' + json.dumps(dict(call=self.calls,
              checkpoint_sha256=self.checkpoint_sha256, input_sha256=key,
              tail_shape=list(self.cache['tail'].shape), predicted_length=self.cache['predicted_length'],
              uses_ground_truth_caption=False, final_adapter_projection_executed=True,
              legacy_clip_complete_chain=self.legacy_clip,
              coordinate_policy=self.cache.get('coordinate_policy', 'native_raw_token_prediction'))), flush=True)
        return self.cache['tail']


def student_text_embedder(checkpoint, legacy_clip=False):
    # Imported only in the H3 environment; the subprocess uses the music environment.
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PromptEmbedder
    from music_text_condition import prefix_length, replace_tail

    class StudentTextTailEmbedder(MiniMaxH3Unit_PromptEmbedder):
        def __init__(self):
            super().__init__()
            self.proxy = MusicTextEncodingProxy(checkpoint, legacy_clip=legacy_clip)
            self.waveform = None

        def set_waveform(self, waveform):
            if waveform.shape != (2, 162000):
                raise ValueError('Student text path requires the shared fixed-length32k waveform')
            self.waveform = waveform

        def process(self, pipe, prompt, keyframes=None, ref_blocks=None, height=None, width=None, text_embedding=None):
            if prompt != ' ' or text_embedding is not None or keyframes or self.waveform is None or not ref_blocks:
                raise ValueError('Student must have only music and reference structure, no semantic prompt/GT tokens')
            encoded = super().process(pipe, prompt, ref_blocks=ref_blocks, height=height, width=width)
            input_ids = self.preprocess_ref_blocks(pipe, prompt, ref_blocks)[0]
            lp = prefix_length(pipe.tokenizer, prompt, input_ids)
            hidden = encoded['prompt_embeds']
            # The isolated music process shares this GPU; release the completed text encoder first.
            context = hidden[lp - 1].detach().cpu()
            pipe.load_models_to_device([])
            tail = self.proxy(self.waveform, context)
            return replace_tail(hidden[:lp], encoded['text_token_tags'][:lp], tail)

    return StudentTextTailEmbedder()


def layout_audit_main():
    from types import SimpleNamespace
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder
    from music_text_condition import replace_tail, write_new, tensor_sha256
    parser = argparse.ArgumentParser(description='Shared student tail and actual H3 text/packed layout audit')
    parser.add_argument('--layout_audit', action='store_true', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--reference_target', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not args.out.parent.is_dir():
        parser.error('Require a new layout audit report')
    probe = torch.load(args.probe, map_location='cpu', weights_only=True)
    prefix_record = torch.load(args.prefix, map_location='cpu', weights_only=True)
    target = torch.load(args.reference_target, map_location='cpu', weights_only=True)
    if probe['checkpoint_sha256'] != sha256(args.checkpoint):
        raise ValueError('Probe checkpoint changed')
    if not torch.equal(prefix_record['context'].float(), probe['context'].float()):
        raise ValueError('Actual empty-prefix context differs from training/deployment probe')
    proxy = MusicTextEncodingProxy(args.checkpoint)
    deployed_tail = proxy(probe['waveform'], prefix_record['context'])
    if not torch.equal(probe['tail'], deployed_tail):
        raise ValueError('Training/evaluation tail differs from actual H3 music proxy')
    expected = replace_tail(prefix_record['prefix'], prefix_record['prefix_tags'], probe['tail'])
    actual = replace_tail(prefix_record['prefix'], prefix_record['prefix_tags'], deployed_tail)
    if any(not torch.equal(expected[key], actual[key]) for key in expected):
        raise ValueError('Packed text input is not the same raw student condition')
    width, height = target['metadata']['prepared_image_size']
    ref_blocks = [dict(kind='image', latent_t=1, latent_h=height // 16, latent_w=width // 16, ref_audio_t=0),
                  dict(kind='audio', ref_audio_t=203)]
    video = torch.empty(1, 24, ((124 - 5) // 17) * 5 + 2, 1280 // 16, 704 // 16)
    audio = torch.empty(2, 32, round(124 / 24 * 40))
    builder = MiniMaxH3Unit_PackedSequenceBuilder()
    packed = []
    for condition in (expected, actual):
        packed.append(builder.process(SimpleNamespace(device='cpu'), video_latents=video, audio_latents=audio,
                      ref_blocks=ref_blocks, **condition)['packed'])
    if set(packed[0]) != set(packed[1]) or any(
            not torch.equal(packed[0][key], packed[1][key]) if torch.is_tensor(packed[0][key])
            else packed[0][key] != packed[1][key] for key in packed[0]):
        raise ValueError('Actual H3 PackedSequenceBuilder layout differs')
    report = dict(complete=True, checkpoint_sha256=sha256(args.checkpoint),
        actual_empty_tc_prefix_used=True, student_reads_caption_or_target_length=False,
        proxy_tail_exact=True, prompt_embeds_tags_exact=True, original_packed_builder_exact=True,
        prefix_length=len(prefix_record['prefix']), student_length=len(deployed_tail),
        packed_sha256={key: tensor_sha256(value) for key, value in packed[0].items() if torch.is_tensor(value)},
        reference_prepared_size=[width, height], video_shape=list(video.shape), audio_shape=list(audio.shape),
        scope='Actual frozen prefix plus shared student condition and original H3 packed layout; no DiT prediction loss or video quality claim')
    write_new(args.out, report)
    print(json.dumps(dict(event='h3_layout_audit', **report)), flush=True)


def main():
    if '--layout_audit' in __import__('sys').argv:
        return layout_audit_main()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint_sha256', required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--legacy_clip', action='store_true', help='Complete old encoder/32Dadapter/CLIPhead -> one globally scaled token; diagnostic only')
    args = parser.parse_args()
    if args.out.exists() or not args.out.parent.is_dir() or sha256(args.checkpoint) != args.checkpoint_sha256:
        parser.error('Require a new output and the declared unchanged checkpoint SHA')
    data = torch.load(args.input, map_location='cpu', weights_only=True)
    if set(data) != {'waveform', 'context'}:
        raise ValueError('Inference only accepts music and structural image context, never caption/length')
    waveform, context = data['waveform'], data['context']
    if waveform.shape != (2, 162000) or context.shape != (5120,):
        raise ValueError('Invalid shared-loader waveform or raw context shape')
    if args.legacy_clip:
        from music_encoder_adapter import load_music_checkpoint
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        mean, std = checkpoint['text_mean'].float(), checkpoint['text_std'].float()
        if (mean.shape != (5120,) or std.shape != (5120,) or not torch.isfinite(mean).all()
                or not torch.isfinite(std).all() or torch.any(std <= 0)):
            raise ValueError('Old checkpoint lacks valid train-only text statistics')
        # No inverse L2 reconstruction: one global scalar matches the train pooled-vector expected norm.
        # The direction remains the old standardized CLIP coordinate; not native token equivalence.
        scale = (mean.square().sum() + std.square().sum()).sqrt()
        model = load_music_checkpoint(args.checkpoint).cuda().eval()
        with torch.no_grad():
            embedding = model(waveform.cuda()[None], effective_frames=torch.tensor([203], device='cuda'))
        tail = (embedding[0].cpu() * scale).reshape(1, 5120)
        result = dict(tail=tail, predicted_length=1, query_max=1,
                      checkpoint_sha256=args.checkpoint_sha256,
                      metadata=dict(kind='legacy_full_clip_chain_global_token', scale=float(scale),
                                    normalization_count=checkpoint.get('normalization', {}).get('count')),
                      coordinate_policy='standardized_CLIP_direction_scaled_by_train_pooled_expected_norm_not_inverse_L2',
                      uses_ground_truth_caption=False)
    else:
        model = load_music_text_checkpoint(args.checkpoint, expected_backend='musicfm')
        if not isinstance(model, MusicToH3Text):
            raise ValueError('Deployment requires a complete backbone+adapter checkpoint')
        model = model.cuda().eval()
        with torch.no_grad():
            tails, lengths = model.generate(waveform.cuda()[None], context.cuda()[None])
        result = dict(tail=tails[0].cpu(), predicted_length=int(lengths[0]),
                      query_max=model.adapter.qmax, checkpoint_sha256=args.checkpoint_sha256,
                      metadata=model.checkpoint_metadata, uses_ground_truth_caption=False)
    with args.out.open('xb') as stream:
        torch.save(result, stream)
    print(json.dumps(dict(event='deployed_raw_text', tokens=result['predicted_length'],
                          query_max=result['query_max'], checkpoint_sha256=args.checkpoint_sha256,
                          legacy_clip_complete_chain=args.legacy_clip)), flush=True)


if __name__ == '__main__':
    main()
