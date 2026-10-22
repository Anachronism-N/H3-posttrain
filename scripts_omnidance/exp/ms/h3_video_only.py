"""Isolated H3 video-only execution using unchanged pretrained video/text weights."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
from tqdm import tqdm

UPSTREAM = Path('/apdcephfs_gy4/share_302533218/cedricnie/DiffSynth-Studio-upstream-20260930')
sys.path.insert(0, str(UPSTREAM))
from diffsynth.diffusion import FlowMatchScheduler
from diffsynth.diffusion.base_pipeline import BasePipeline, PipelineUnit
from diffsynth.models.minimax_h3_dit import MiniMaxH3DiT, MiniMaxH3FinalLayer, patchify_video, unpatchify_video
from diffsynth.pipelines.minimax_h3_audio_video import (
    MiniMaxH3Pipeline, ModelConfig, MiniMaxH3Unit_ShapeChecker,
    MiniMaxH3Unit_InputVideoEmbedder, MiniMaxH3Unit_VideoRetakeEmbedder,
    MiniMaxH3Unit_KeyframeEncoder, MiniMaxH3Unit_PromptEmbedder,
    MiniMaxH3Unit_ControlNetEncoder, MiniMaxH3Unit_ReferenceEncoder,
    MiniMaxH3Unit_PackedSequenceBuilder,
)

PROTOCOL = 'h3_caption_reference_video_only_v1'


def video_prompt(caption):
    if not isinstance(caption, str) or not caption.strip() or '<Audio' in caption:
        raise ValueError('Require nonempty dance caption without audio-reference directives')
    return (
        'subject_definitions:\n'
        '<Subject 1> is the dancer shown in <Picture 1>, preserving the reference identity, outfit and setting.\n\n'
        'summary:\nA realistic single continuous shot shows <Subject 1> performing the following dance.\n\n'
        'detailed_description:\n[Shot 1] ' + caption.strip()
    )


def _reject_audio(inputs):
    for key in ('input_audio', 'retake_audio', 'input_latents_audio', 'audio_input_latents',
                'denoise_mask_audio', 'ref_audio_anchor'):
        if inputs.get(key) is not None:
            raise ValueError('Video-only forbids audio input: ' + key)
    if inputs.get('audio_latents') is not None and inputs['audio_latents'].numel():
        raise ValueError('Video-only has no nonempty audio latents')
    for ref in inputs.get('references') or []:
        if ref.get('type') not in ('image', 'video') or ref.get('audio') is not None:
            raise ValueError('Video-only permits only silent image/video references')
    for block in inputs.get('ref_blocks') or []:
        if block.get('kind') not in ('image', 'video') or block.get('ref_audio_t', 0):
            raise ValueError('Video-only reference contains audio rows')


class VideoOnlyFinalLayer(MiniMaxH3FinalLayer):
    def forward(self, x, *, t_emb, inverse_indices):
        shift, scale = self.adaln_proj(t_emb)
        h = self.norm(x)
        h = (h * (1 + scale.index_select(0, inverse_indices))
             + shift.index_select(0, inverse_indices)).to(x.dtype)
        return self.video_out(h), h.new_empty((0, self.audio_out.out_features))


class VideoOnlyDiT(MiniMaxH3DiT):
    def _embed(self, *, x, audio_x, text_embeddings_selected, unique_timesteps,
               img_pos, audio_pos, text_pos, refiner_cu_seqlens, refiner_max_seqlen,
               seq_len, device):
        if audio_pos.numel() or (audio_x is not None and audio_x.numel()):
            raise ValueError('Video-only DiT cannot embed audio')
        dtype = text_embeddings_selected.dtype
        video_rows = x.view(-1, x.shape[-1]).index_select(0, img_pos).to(dtype)
        video_embed = self.video_patch_proj(video_rows)
        text_embed = self.condition_proj(text_embeddings_selected.to(device=device))
        text_embed = self.token_refiner(text_embed, cu_seqlens=refiner_cu_seqlens,
                                       max_seqlen=refiner_max_seqlen)
        embedding = torch.zeros((seq_len, self.hidden_size), device=device, dtype=dtype)
        embedding.index_add_(0, text_pos, text_embed.to(dtype))
        embedding.index_add_(0, img_pos, video_embed.to(dtype))
        return embedding, self.time_embedder(unique_timesteps, dtype=dtype)


def enable_video_only_dit(dit):
    """Change execution only; state names/shapes and old AV source files remain intact."""
    targets = [m for m in dit.modules() if type(m) is MiniMaxH3DiT]
    if isinstance(dit, VideoOnlyDiT):
        return dit
    if len(targets) != 1:
        raise ValueError('Require exactly one pretrained H3 DiT, not a different architecture')
    model = targets[0]
    finals = [m for m in model.final_layer.modules() if type(m) is MiniMaxH3FinalLayer]
    if len(finals) != 1:
        raise ValueError('Require one original final layer')
    model.__class__ = VideoOnlyDiT
    finals[0].__class__ = VideoOnlyFinalLayer
    model.audio_patch_proj.requires_grad_(False)
    finals[0].audio_out.requires_grad_(False)
    return dit


class VideoOnlyNoise(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=('seed', 'num_frames', 'height', 'width', 'rand_device'),
                         output_params=('video_latents', 'audio_latents'))

    def process(self, pipe, seed, num_frames, height, width, rand_device):
        frames = ((num_frames - 5) // 17) * 5 + 2
        video = pipe.generate_noise((1, 24, frames, height // 16, width // 16), seed=seed,
                                    rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        return dict(video_latents=video, audio_latents=video.new_empty((2, 32, 0)))


class VideoOnlyReference(MiniMaxH3Unit_ReferenceEncoder):
    def __init__(self):
        super().__init__()
        self.onload_model_names = ('video_vae',)

    def process(self, pipe, references, **kwargs):
        _reject_audio(dict(references=references))
        result = super().process(pipe, references=references, **kwargs)
        if result.get('ref_audio_anchor') is not None:
            raise ValueError('Audio anchor unexpectedly encoded')
        return result


class VideoOnlyPacked(MiniMaxH3Unit_PackedSequenceBuilder):
    def process(self, pipe, prompt_embeds, video_latents, audio_latents=None, **kwargs):
        _reject_audio(dict(audio_latents=audio_latents, **kwargs))
        if audio_latents is None:
            audio_latents = video_latents.new_empty((2, 32, 0))
        result = super().process(pipe, prompt_embeds=prompt_embeds, video_latents=video_latents,
                                 audio_latents=audio_latents, **kwargs)
        packed = result['packed']
        if packed['audio_pos'].numel() or (packed['token_tags'] == 2).any():
            raise ValueError('Packed sequence contains audio tokens')
        return result


def video_only_model_fn(dit, video_latents, packed, prompt_embeds, timestep_video,
                        audio_latents=None, timestep_audio=None, keyframe_cond_anchor=None,
                        ref_visual_anchor=None, input_latents_video=None, denoise_mask_video=None,
                        imgvid_cond_noise_aug=0.999, use_gradient_checkpointing=False,
                        use_gradient_checkpointing_offload=False, **kwargs):
    _reject_audio(dict(audio_latents=audio_latents, **kwargs))
    if timestep_audio is not None:
        raise ValueError('Video-only has no audio diffusion timestep')
    if packed['audio_pos'].numel() or (packed['token_tags'] == 2).any():
        raise ValueError('No audio positions/modalities allowed in video-only forward')
    dtype, device = video_latents.dtype, video_latents.device
    frames, height, width = video_latents.shape[2:]
    t_video = 1.0 - float(timestep_video) / 1000
    if input_latents_video is not None:
        video_latents = video_latents * denoise_mask_video + input_latents_video * (1 - denoise_mask_video)
        mask = patchify_video(denoise_mask_video)[:, 0]
    video_rows = patchify_video(video_latents)
    image_pos, text_pos = packed['img_pos'], packed['text_pos']
    anchor = ref_visual_anchor if ref_visual_anchor is not None else keyframe_cond_anchor
    reference_rows = 0 if anchor is None else len(anchor)
    x = torch.zeros(1, packed['seq_len'], 96, dtype=dtype, device=device)
    x[0].index_copy_(0, image_pos[reference_rows:], video_rows)
    if anchor is not None:
        x[0].index_copy_(0, image_pos[:reference_rows], anchor)
    timesteps = torch.full((packed['seq_len'],), t_video, dtype=torch.float32, device=device)
    if input_latents_video is not None:
        timesteps[image_pos[reference_rows:][mask == 0]] = 1
    timesteps[image_pos[:reference_rows]] = max(t_video, imgvid_cond_noise_aug)
    unique, inverse = torch.unique(timesteps, sorted=True, return_inverse=True)
    text_len = len(text_pos)
    refiner_cu = torch.tensor([0, text_len, text_len], dtype=torch.int32, device=device)
    prediction, audio_prediction = dit(
        x=x, audio_x=None, img_position_ids=packed['img_position_ids'], unique_timesteps=unique,
        inverse_indices=inverse, token_tags=packed['token_tags'], prompt_embeds=prompt_embeds,
        img_pos_info={'position_ids': image_pos}, audio_pos_info={'position_ids': packed['audio_pos']},
        text_pos_info={'position_ids': text_pos}, img_pos_for_infer_output_info={'position_ids': image_pos},
        packed_seq_params={'cu_seqlens_q': packed['cu_seqlens'], 'max_seqlen_q': int(packed['cu_seqlens'][1])},
        refiner_packed_seq_params={'cu_seqlens_q': refiner_cu, 'max_seqlen_q': text_len},
        update_mask=None, skip_mask_out_condition=True,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload)
    if audio_prediction.numel():
        raise ValueError('Video-only unexpectedly predicted audio')
    video = -unpatchify_video(prediction[reference_rows:], frames, height, width)
    return video, video.new_empty((2, 32, 0))


class H3VideoOnlyPipeline(MiniMaxH3Pipeline):
    def __init__(self, device='cpu', torch_dtype=torch.bfloat16):
        BasePipeline.__init__(self, device=device, torch_dtype=torch_dtype,
                              height_division_factor=32, width_division_factor=32,
                              time_division_factor=17, time_division_remainder=5)
        self.scheduler = FlowMatchScheduler('MiniMax-H3')
        self.scheduler_audio = None
        self.text_encoder = self.dit = self.video_vae = self.audio_vae = self.controlnet = None
        self.tokenizer = self.processor = None
        self.imgvid_cond_noise_aug, self.audio_cond_noise_aug = 0.999, 1.
        self.in_iteration_models = ('dit',)
        self.units = [MiniMaxH3Unit_ShapeChecker(), VideoOnlyNoise(),
                      MiniMaxH3Unit_InputVideoEmbedder(), MiniMaxH3Unit_VideoRetakeEmbedder(),
                      MiniMaxH3Unit_KeyframeEncoder(), VideoOnlyReference(),
                      MiniMaxH3Unit_PromptEmbedder(), VideoOnlyPacked(),
                      MiniMaxH3Unit_ControlNetEncoder()]
        self.model_fn = video_only_model_fn
        self.compilable_models = ['dit']
        from diffsynth.utils.lora.minimax_h3 import MiniMaxH3LoRALoader
        self.lora_loader = MiniMaxH3LoRALoader

    @torch.no_grad()
    def __call__(self, prompt=None, negative_prompt=None, height=1280, width=704, num_frames=124,
                 num_inference_steps=50, seed=42, rand_device='cpu', cfg_scale=1., flow_shift=12.,
                 tiled=True, tile_size=256, tile_overlap=64, references=None, text_embedding=None,
                 ref_image_short_edge=2048, ref_video_short_edge=768, ref_video_max_pixels=768 * 1344,
                 use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False,
                 progress_bar_cmd=tqdm, **forbidden):
        if forbidden:
            raise ValueError('Unsupported video-only inputs: ' + ', '.join(sorted(forbidden)))
        _reject_audio(dict(references=references))
        if cfg_scale != 1:
            raise ValueError('First video-only experiment preserves distilled CFG1')
        if self.audio_vae is not None or self.scheduler_audio is not None:
            raise ValueError('Audio modules must be absent')
        if prompt is not None and ('<Audio' in prompt or 'overall_soundscape:' in prompt):
            raise ValueError('Video-only prompt contains audio task directives')
        self.scheduler.set_timesteps(num_inference_steps, shift=flow_shift)
        positive, negative = {'prompt': prompt}, {'negative_prompt': negative_prompt}
        shared = dict(cfg_scale=cfg_scale, height=height, width=width, num_frames=num_frames, seed=seed,
                      rand_device=rand_device, tiled=tiled, tile_size=tile_size, tile_overlap=tile_overlap,
                      use_gradient_checkpointing=use_gradient_checkpointing,
                      use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                      keyframes=None, keyframe_indices=None, references=references,
                      ref_image_short_edge=ref_image_short_edge, ref_video_short_edge=ref_video_short_edge,
                      ref_video_max_pixels=ref_video_max_pixels, text_embedding=text_embedding,
                      imgvid_cond_noise_aug=self.imgvid_cond_noise_aug, audio_cond_noise_aug=1.,
                      input_video=None, retake_video=None, frame_regions_to_retake=None,
                      control_video=None, control_scale=1.)
        for unit in self.units:
            shared, positive, negative = self.unit_runner(unit, self, shared, positive, negative)
        self.load_models_to_device(self.in_iteration_models)
        for index, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            prediction, audio = self.model_fn(dit=self.dit, **shared, **positive,
                timestep_video=timestep.unsqueeze(0).to(dtype=torch.float32, device=self.device))
            if audio.numel():
                raise ValueError('Unexpected audio prediction')
            shared['video_latents'] = self.step(self.scheduler, shared['video_latents'], index,
                noise_pred=prediction, inpaint_mask=shared.get('denoise_mask_video'),
                input_latents=shared.get('input_latents_video'))
        self.load_models_to_device(['video_vae'])
        frames = self.video_vae.decode_video(shared['video_latents'], dtype=self.torch_dtype,
                                            tiled=tiled, tile_size=tile_size, tile_overlap=tile_overlap)
        return self.vae_output_to_video(frames, min_value=0, max_value=1)


def build_video_only_pipeline(weights_root, device='cpu', torch_dtype=torch.bfloat16,
                              load_text_encoder=False, vram_limit=None, load_mode='all',
                              model_configs=None, vram_management=False):
    if load_mode not in ('all', 'cache', 'data_process', 'train'):
        raise ValueError('Unknown video-only model loading stage')
    weights = Path(weights_root)
    configurations = []
    memory = dict(offload_dtype=torch_dtype, offload_device='cpu', onload_dtype=torch_dtype,
                  onload_device='cpu', preparing_dtype=torch_dtype, preparing_device=device,
                  computation_dtype=torch_dtype, computation_device=device) if vram_management else {}
    if model_configs is not None:
        for configuration in model_configs:
            if 'audio_vae' in str(configuration.path):
                raise ValueError('Video-only must not load audio VAE weights')
        configurations = model_configs
    else:
        if load_mode in ('all', 'train'):
            paths = sorted(str(p) for p in (weights / 'transformer').glob('model*.safetensors'))
            if not paths:
                raise FileNotFoundError(weights / 'transformer')
            configurations.append(ModelConfig(path=paths, **memory))
        if load_mode in ('all', 'cache', 'data_process'):
            configurations.append(ModelConfig(path=str(weights / 'video_vae/source/model.safetensors'), **memory))
        if load_text_encoder or load_mode in ('cache', 'data_process'):
            paths = sorted(str(p) for p in (weights / 'text_encoder').glob('model*.safetensors'))
            if not paths:
                raise FileNotFoundError(weights / 'text_encoder')
            configurations.append(ModelConfig(path=paths, **memory))
    pipe = H3VideoOnlyPipeline(device=device, torch_dtype=torch_dtype)
    pool = pipe.download_and_load_models(configurations, vram_limit)
    pipe.dit = pool.fetch_model('minimax_h3_dit')
    pipe.video_vae = pool.fetch_model('minimax_h3_video_vae')
    pipe.text_encoder = pool.fetch_model('minimax_h3_text_encoder')
    if pipe.dit is not None:
        enable_video_only_dit(pipe.dit)
    if load_text_encoder or load_mode in ('cache', 'data_process'):
        from transformers import AutoProcessor
        pipe.processor = AutoProcessor.from_pretrained(str(weights / 'processor'), local_files_only=True)
        pipe.tokenizer = pipe.processor.tokenizer
    pipe.requires_grad_(False).eval()
    pipe.vram_management_enabled = pipe.check_vram_management_state()
    return pipe


def self_test():
    from diffsynth.pipelines.minimax_h3_audio_video import model_fn_minimax_h3
    torch.manual_seed(42)
    base = MiniMaxH3DiT(num_layers=2, token_refiner_num_layers=1, hidden_size=48,
        num_attention_heads=4, attention_head_dim=12, ffn_hidden_size=64, time_embed_dim=32,
        time_embed_hidden_size=48, adaln_out_features=864, final_adaln_out_features=96,
        rope_inv_freq_len=2).float()
    model = copy.deepcopy(base)
    state_before = {k: v.clone() for k, v in model.state_dict().items()}
    enable_video_only_dit(model)
    assert all(torch.equal(state_before[k], v) for k, v in model.state_dict().items())
    from unittest.mock import patch
    scheduler_calls, real_scheduler = [], FlowMatchScheduler
    def make_scheduler(*args, **kwargs):
        scheduler_calls.append(args)
        return real_scheduler(*args, **kwargs)
    with patch(__name__ + '.FlowMatchScheduler', side_effect=make_scheduler):
        pipe = H3VideoOnlyPipeline(torch_dtype=torch.float32)
    assert len(scheduler_calls) == 1
    video = torch.randn(1, 24, 2, 4, 4)
    text = torch.randn(3, 5120, requires_grad=True)
    empty = torch.empty(2, 32, 0)
    reference = torch.randn(4, 96)
    refs = [dict(kind='image', latent_t=1, latent_h=4, latent_w=4, ref_audio_t=0)]
    packed = VideoOnlyPacked().process(pipe, prompt_embeds=text, video_latents=video,
        audio_latents=empty, text_token_tags=torch.ones(3, dtype=torch.long), ref_blocks=refs)['packed']
    common = dict(video_latents=video, audio_latents=empty, packed=packed, prompt_embeds=text,
                  timestep_video=torch.tensor([500.]), ref_visual_anchor=reference)
    original, _ = model_fn_minimax_h3(dit=base, timestep_audio=torch.tensor([500.]), **common)
    calls = []
    def forbidden(module, args):
        calls.append(type(module).__name__)
        raise AssertionError('Audio projection/head executed')
    handles = [model.audio_patch_proj.register_forward_pre_hook(forbidden),
               model.final_layer.audio_out.register_forward_pre_hook(forbidden)]
    prediction, audio = video_only_model_fn(dit=model, **common)
    assert prediction.shape == video.shape and audio.numel() == 0
    assert torch.equal(prediction, original)
    prediction.square().mean().backward()
    assert text.grad is not None and torch.isfinite(text.grad).all() and text.grad.norm() > 0
    assert any(p.grad is not None for p in model.blocks.parameters())
    assert all(p.grad is None for p in model.audio_patch_proj.parameters())
    assert all(p.grad is None for p in model.final_layer.audio_out.parameters())
    assert not calls and pipe.audio_vae is None and pipe.scheduler_audio is None
    class TestVideoVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def decode_video(self, latents, **kwargs):
            self.calls += 1
            return torch.zeros(1, 3, 5, 32, 32)
    pipe.dit, pipe.video_vae = model, TestVideoVAE()
    generated = pipe(text_embedding=text.detach(), height=32, width=32, num_frames=5,
                     num_inference_steps=2, progress_bar_cmd=lambda values: values)
    assert len(generated) == 5 and generated[0].size == (32, 32)
    assert pipe.video_vae.calls == 1 and not calls
    for handle in handles:
        handle.remove()
    rejected = []
    for name, inputs in (('nonempty_audio', dict(audio_latents=torch.ones(2, 32, 1))),
                         ('audio_reference', dict(references=[dict(type='audio')])),
                         ('video_audio_reference', dict(ref_blocks=[dict(kind='video_audio')])),
                         ('audio_target', dict(audio_input_latents=torch.ones(2, 32, 1)))):
        try:
            _reject_audio(inputs)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError('Audio input accepted')
    assert len(rejected) == 4
    return dict(complete=True, cpu_only=True, real_tiny_H3_blocks=True,
                full_pretrained_weights_loaded=False, video_only_matches_AV_zero_length_audio=True,
                unchanged_state_dict=True, real_input_and_decoder_gradients=True,
                audio_projection_head_calls=0, audio_scheduler_and_vae_absent=True,
                scheduler_initializations=1, complete_video_only_sampling_test=True,
                tiny_video_scheduler_updates=2, tiny_video_vae_decode_calls=1,
                rejected=rejected, scope='Actual small H3 architecture and fake video decode; no large-model GPU/video quality claim')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if not args.self_test:
        parser.error('Use the task-specific training or generation driver')
    print(json.dumps(self_test()), flush=True)
