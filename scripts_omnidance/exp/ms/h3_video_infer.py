"""Audited caption + reference-image H3 video-only condition cache and inference.

No audio media loader, audio VAE or audio scheduler is used. Review music can be
named in an opaque sidecar field but is never opened or passed to H3.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

PROTOCOL = 'h3_caption_image_infer_v1'
CACHE_FORMAT = 'h3_new_caption_image_condition_v1'
HEIGHT, WIDTH, FRAMES, FPS = 1280, 704, 124, 24
FORBIDDEN_CAPTION = re.compile(r'<\s*Audio\b|overall_soundscape\s*:|non_diegetic_music\s*:|audio reuse|fully_copy|soundtrack', re.I)
FORBIDDEN_KEYS = {'audio', 'audios', 'audio_path', 'input_audio', 'retake_audio', 'references',
                  'input_video', 'keyframes', 'text_embedding', 'prompt', 'h3_prompt'}


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f'Duplicate JSON key: {key}')
        out[key] = value
    return out


def read_json(path):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream, object_pairs_hook=unique_object)


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def source(path):
    path = Path(path).absolute()
    before, digest = path.stat(), hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'Source changed while hashing: {path}')
    return dict(path=str(path), sha256=digest.hexdigest(), size_bytes=after.st_size)


def check_new(path):
    path = Path(path).absolute()
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    if os.path.lexists(path):
        raise FileExistsError(path)
    return path


def atomic_text(path, text):
    path = check_new(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')


def atomic_torch(path, value):
    import torch
    path = check_new(path)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        with open(temporary, 'rb') as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def captions(path, limit):
    value = read_json(path)
    records = value if isinstance(value, list) else value.get('records') if isinstance(value, dict) else None
    if records is None and isinstance(value, dict) and 'id' in value:
        records = [value]
    if not isinstance(records, list) or not records:
        raise ValueError('Caption JSON requires one explicit record or a records list')
    ids = []
    for record in records:
        if not isinstance(record, dict) or FORBIDDEN_KEYS & set(record):
            raise ValueError('Caption inference rows may not contain old AV/prompt/input fields')
        sid = record.get('id')
        if not isinstance(sid, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', sid):
            raise ValueError('Each caption record requires a safe explicit ID')
        for key in ('image_path', 'motion_caption', 'caption_version'):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f'Missing {key}: {sid}')
        if FORBIDDEN_CAPTION.search(record['motion_caption']):
            raise ValueError(f'Old AV/audio directive in motion_caption: {sid}')
        if any(marker in record['motion_caption'] for marker in ('subject_definitions:', 'detailed_description:', '<Picture', '<Subject')):
            raise ValueError(f'motion_caption must be raw unwrapped prose: {sid}')
        if not Path(record['image_path']).is_file():
            raise FileNotFoundError(record['image_path'])
        ids.append(sid)
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate inference IDs')
    if not 1 <= limit <= 3 or limit > len(records):
        raise ValueError('--limit must select one early case or at most the fixed three cases')
    return records[:limit]


def weight_sources(root, cache_only, from_cache):
    root = Path(root)
    directories = ['video_vae']
    if not cache_only:
        directories.append('transformer')
    if not from_cache:
        directories += ['text_encoder', 'processor']
    result = []
    for name in directories:
        directory = root / name
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        files = sorted({*directory.rglob('*.safetensors'), *directory.rglob('*.json'),
                        *directory.rglob('*.txt'), *directory.rglob('*.jinja')})
        if not files or (name != 'processor' and not any(path.suffix == '.safetensors' for path in files)):
            raise ValueError(f'No complete model/config sources: {directory}')
        result.extend(source(path) for path in files)
    return dict(weights_root=str(root.absolute()), files=result,
                load_scope=directories, sha256=object_sha(result), audio_vae_loaded=False)


def tree_to(value, device=None):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu() if device is None else value.to(device)
    if isinstance(value, dict):
        return {key: tree_to(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_to(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(tree_to(item, device) for item in value)
    return value


def packed_audit(packed):
    import torch
    for key in ('audio_pos', 'token_tags', 'img_pos', 'text_pos'):
        if key not in packed or not isinstance(packed[key], torch.Tensor):
            raise ValueError(f'Missing real packed tensor: {key}')
    audio_positions = packed['audio_pos'].numel()
    tag2 = int((packed['token_tags'] == 2).sum().item())
    if audio_positions or tag2:
        raise AssertionError('Audio positions/tag2 in video-only condition')
    return dict(audio_pos_numel=audio_positions, tag2_count=tag2,
                video_position_rows=packed['img_pos'].numel(), text_position_rows=packed['text_pos'].numel(),
                token_tag_counts={str(int(tag)): int((packed['token_tags'] == tag).sum())
                                  for tag in packed['token_tags'].unique().cpu().tolist()})


class AudioGuard:
    def __init__(self, pipe):
        if pipe.audio_vae is not None or pipe.scheduler_audio is not None:
            raise AssertionError('Audio VAE/scheduler must permanently be None')
        self.pipe = pipe
        self.calls = dict(audio_patch=0, audio_head=0, audio_reference_encode=0)
        self.handles, self.replaced, self.observed_methods = [], [], []
        self.video_calls = dict(text_encoder=0, video_vae_encode=0, video_vae_decode=0, dit=0)
        for name, module in [('text_encoder', pipe.text_encoder), ('dit', pipe.dit)]:
            if module is not None:
                def observed(module, args, name=name):
                    self.video_calls[name] += 1
                self.handles.append(module.register_forward_pre_hook(observed))
        if pipe.video_vae is not None:
            for name, method_name in [('video_vae_encode', 'encode_video'), ('video_vae_decode', 'decode_video')]:
                if hasattr(pipe.video_vae, method_name):
                    old = getattr(pipe.video_vae, method_name)
                    def observed_method(*args, old=old, name=name, **kwargs):
                        self.video_calls[name] += 1
                        return old(*args, **kwargs)
                    setattr(pipe.video_vae, method_name, observed_method)
                    self.observed_methods.append((pipe.video_vae, method_name, old))
        self.audio_hook_count = 0
        if pipe.dit is not None:
            for name, module in pipe.dit.named_modules():
                category = 'audio_patch' if name.endswith('audio_patch_proj') else 'audio_head' if name.endswith('audio_out') else None
                if category is not None:
                    def forbidden(module, args, category=category):
                        self.calls[category] += 1
                        raise AssertionError(f'Forbidden {category} forward')
                    self.handles.append(module.register_forward_pre_hook(forbidden))
                    self.audio_hook_count += 1
            if self.audio_hook_count < 2:
                raise AssertionError('Audio projection/head hook coverage missing')
        for unit in pipe.units:
            if hasattr(unit, '_encode_audio_ref'):
                old = unit._encode_audio_ref
                def forbidden_audio(*args, **kwargs):
                    self.calls['audio_reference_encode'] += 1
                    raise AssertionError('Audio waveform/reference encoding forbidden')
                unit._encode_audio_ref = forbidden_audio
                self.replaced.append((unit, old))

    def report(self):
        if self.pipe.audio_vae is not None or self.pipe.scheduler_audio is not None or any(self.calls.values()):
            raise AssertionError('Video-only audio exclusion violated')
        return dict(calls=dict(self.calls), installed_projection_head_hooks=self.audio_hook_count,
                    observed_video_calls=dict(self.video_calls), audio_vae_is_none=True, scheduler_audio_is_none=True)

    def close(self):
        for handle in self.handles:
            handle.remove()
        for unit, old in self.replaced:
            unit._encode_audio_ref = old
        for module, method_name, old in self.observed_methods:
            setattr(module, method_name, old)


def prepare_condition(pipe, record, geometry, seed):
    from PIL import Image
    from h3_video_only import video_prompt
    with Image.open(record['image_path']) as image:
        reference = image.convert('RGB').copy()
    prompt = video_prompt(record['motion_caption'])
    positive, negative = {'prompt': prompt}, {}
    shared = dict(cfg_scale=1., height=geometry['height'], width=geometry['width'],
                  num_frames=geometry['num_frames'], seed=seed, rand_device='cpu', tiled=True,
                  tile_size=256, tile_overlap=64, use_gradient_checkpointing=False,
                  use_gradient_checkpointing_offload=False, keyframes=None, keyframe_indices=None,
                  references=[dict(type='image', image=reference)], ref_image_short_edge=2048,
                  ref_video_short_edge=768, ref_video_max_pixels=768 * 1344, text_embedding=None,
                  imgvid_cond_noise_aug=pipe.imgvid_cond_noise_aug, audio_cond_noise_aug=1.,
                  input_video=None, retake_video=None, frame_regions_to_retake=None,
                  control_video=None, control_scale=1.)
    text_calls = []
    handle = pipe.text_encoder.register_forward_pre_hook(lambda module, args: text_calls.append(1))
    try:
        for unit in pipe.units:
            shared, positive, negative = pipe.unit_runner(unit, pipe, shared, positive, negative)
    finally:
        handle.remove()
    if len(text_calls) != 1:
        raise AssertionError(f'Expected exactly one positive native TC call, got {len(text_calls)}')
    negative.clear()  # Native CFG1 aliases the positive outputs; it does not encode a negative prompt.
    if shared.get('ref_audio_anchor') is not None or shared['audio_latents'].numel():
        raise AssertionError('Condition encoder created audio')
    audit = packed_audit(positive.get('packed', shared.get('packed')))
    if 'prompt_embeds' not in positive or 'text_token_tags' not in positive:
        raise ValueError('New native TC embedding/text tags missing')
    blocks = []
    for block in shared.get('ref_blocks', []):
        if block['kind'] != 'image' or block.get('ref_audio_t', 0):
            raise AssertionError('Non-image reference block in condition cache')
        metadata = {key: block[key] for key in ('kind', 'latent_t', 'latent_h', 'latent_w', 'ref_audio_t')}
        import numpy as np
        import torch
        prepared = np.asarray(block['prepared_image'].convert('RGB'))
        metadata['prepared_image_rgb'] = torch.from_numpy(prepared.copy())
        blocks.append(metadata)
    payload = dict(ref_visual_anchor=shared['ref_visual_anchor'], ref_blocks=blocks,
                   prompt_embeds=positive['prompt_embeds'], text_token_tags=positive['text_token_tags'],
                   packed=positive.get('packed', shared.get('packed')),
                   video_latent_shape=list(shared['video_latents'].shape))
    return payload, prompt, audit


def cache_envelope(record, condition, prompt, geometry, seed, inputs, models):
    return dict(format=CACHE_FORMAT, protocol=PROTOCOL, id=record['id'],
                motion_caption=record['motion_caption'], caption_version=record['caption_version'],
                prompt=prompt, caption_sha256=object_sha(record['motion_caption']),
                image_source=inputs['image'], caption_json_source=inputs['caption_json'],
                model_sources=models, geometry=geometry, seed=seed,
                wrapper_source=source(Path(__file__).parent / 'h3_video_only.py'),
                source_code=source(__file__), condition=tree_to(condition), old_y_reused=False)


def load_condition(path, record, inputs, geometry, seed, device):
    import torch
    from h3_video_only import video_prompt
    value = torch.load(path, map_location='cpu', weights_only=True)
    if value.get('format') != CACHE_FORMAT or value.get('protocol') != PROTOCOL:
        raise ValueError('Old caption/y or training latent caches forbidden; use this new condition format')
    expected = dict(id=record['id'], caption_version=record['caption_version'],
                    caption_sha256=object_sha(record['motion_caption']), prompt=video_prompt(record['motion_caption']),
                    geometry=geometry, seed=seed)
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ValueError(f'Condition cache mismatch: {key}')
    if value['image_source']['sha256'] != inputs['image']['sha256'] or value['caption_json_source']['sha256'] != inputs['caption_json']['sha256']:
        raise ValueError('Condition cache source SHA mismatch')
    if value['wrapper_source']['sha256'] != source(Path(__file__).parent / 'h3_video_only.py')['sha256']:
        raise ValueError('Condition cache video-only implementation SHA mismatch')
    packed_audit(value['condition']['packed'])
    return tree_to(value['condition'], device), value


def sample_cached(pipe, condition, steps, seed, flow_shift):
    import torch
    packed = packed_audit(condition['packed'])
    pipe.scheduler.set_timesteps(steps, shift=flow_shift)
    latents = pipe.generate_noise(tuple(condition['video_latent_shape']), seed=seed,
                                  rand_device='cpu', rand_torch_dtype=pipe.torch_dtype)
    values = {key: condition[key] for key in ('prompt_embeds', 'text_token_tags', 'packed', 'ref_visual_anchor', 'ref_blocks')}
    values.update(audio_latents=latents.new_empty((2, 32, 0)), cfg_scale=1.,
                  imgvid_cond_noise_aug=pipe.imgvid_cond_noise_aug,
                  use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False)
    pipe.load_models_to_device(('dit',))
    updates = 0
    with torch.inference_mode():
        for index, timestep in enumerate(pipe.scheduler.timesteps):
            prediction, empty_audio = pipe.model_fn(dit=pipe.dit, video_latents=latents, **values,
                                                    timestep_video=timestep.unsqueeze(0).to(dtype=torch.float32, device=pipe.device))
            if empty_audio.numel() or prediction.shape != latents.shape:
                raise AssertionError('Invalid video-only model prediction')
            latents = pipe.step(pipe.scheduler, latents, index, noise_pred=prediction)
            updates += 1
        pipe.load_models_to_device(['video_vae'])
        decoded = pipe.video_vae.decode_video(latents, dtype=pipe.torch_dtype,
                                              tiled=True, tile_size=256, tile_overlap=64)
        frames = pipe.vae_output_to_video(decoded, min_value=0, max_value=1)
    if updates != steps:
        raise AssertionError('Video scheduler update count differs from requested steps')
    return frames, dict(video_scheduler_updates=updates, packed=packed, target_audio_generated=False)


def ffmpeg_binary():
    value = shutil.which('ffmpeg')
    if value:
        return value
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def write_video(path, frames, geometry):
    import numpy as np
    path = check_new(path)
    partial = check_new(path.with_name(path.stem + '.partial.mp4'))
    command = [ffmpeg_binary(), '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f"{geometry['width']}x{geometry['height']}",
               '-r', str(geometry['fps']), '-i', 'pipe:0', '-map', '0:v:0', '-an', '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart', str(partial)]
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr)
        count = 0
        try:
            for frame in frames:
                image = np.asarray(frame)
                if image.shape != (geometry['height'], geometry['width'], 3) or image.dtype != np.uint8:
                    raise ValueError(f'Output frame shape/dtype invalid: {image.shape}/{image.dtype}')
                process.stdin.write(image.tobytes())
                count += 1
            process.stdin.close()
            status = process.wait()
        except BaseException:
            process.stdin.close()
            process.wait()
            raise
        stderr.seek(0)
        log = stderr.read().decode(errors='replace')
    if status or count != geometry['num_frames']:
        raise RuntimeError(f'Video encode failed/count mismatch; partial retained: count={count}, status={status}, log={log}')
    verification = verify_media(partial, geometry)
    os.link(partial, path)
    partial.unlink()
    return verification


def verify_media(path, geometry):
    probe = shutil.which('ffprobe')
    if probe:
        command = [probe, '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', str(path)]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        if result.stderr.strip():
            raise RuntimeError('ffprobe full-decode warning/error: ' + result.stderr)
        streams = json.loads(result.stdout)['streams']
        video = [stream for stream in streams if stream['codec_type'] == 'video']
        audio = [stream for stream in streams if stream['codec_type'] == 'audio']
        if len(video) != 1 or audio or len(streams) != 1:
            raise ValueError('Expected exactly one video stream and no audio/other streams')
        stream = video[0]
        numerator, denominator = [int(part) for part in stream['avg_frame_rate'].split('/')]
        rate = numerator / denominator
        count = int(stream['nb_read_frames'])
        dimensions = (int(stream['height']), int(stream['width']))
        method = 'ffprobe -count_frames: actual stream inspection and complete decoded frame count'
    else:
        inspection = subprocess.run([ffmpeg_binary(), '-nostdin', '-hide_banner', '-i', str(path)], capture_output=True, text=True)
        lines = [line for line in inspection.stderr.splitlines() if re.search(r'^\s*Stream #\d+:\d+', line)]
        if len(lines) != 1 or 'Video:' not in lines[0] or 'Audio:' in inspection.stderr or 'Input #0' not in inspection.stderr:
            raise ValueError('Full input stream inspection did not prove exactly one video/no audio stream')
        frame_bytes = geometry['height'] * geometry['width'] * 3
        command = [ffmpeg_binary(), '-nostdin', '-hide_banner', '-loglevel', 'info', '-xerror',
                   '-i', str(path), '-map', '0:v:0', '-vf', 'showinfo', '-vsync', '0',
                   '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1']
        with tempfile.TemporaryFile() as stderr:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr)
            byte_count = 0
            for chunk in iter(lambda: process.stdout.read(1024 * 1024), b''):
                byte_count += len(chunk)
            process.stdout.close()
            status = process.wait()
            stderr.seek(0)
            decode_log = stderr.read().decode(errors='replace')
        if status:
            raise RuntimeError(f'ffmpeg -xerror complete decode failed: {decode_log}')
        expected_bytes = geometry['num_frames'] * frame_bytes
        if byte_count != expected_bytes:
            raise ValueError(f'Decoded RGB byte count mismatch: {byte_count} != {expected_bytes}')
        events = re.findall(r'\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:\s*([-+\d.eE]+).*?\bs:(\d+)x(\d+)\b', decode_log)
        if len(events) != geometry['num_frames']:
            raise ValueError('showinfo does not prove every decoded frame')
        times, dimensions = [], None
        for index, (number, pts, stamp, width, height) in enumerate(events):
            if int(number) != index or (int(height), int(width)) != (geometry['height'], geometry['width']):
                raise ValueError('showinfo frame index/shape mismatch')
            times.append(float(stamp))
            dimensions = (int(height), int(width))
        if any(not math.isclose(stamp - times[0], index / geometry['fps'], rel_tol=0, abs_tol=1e-5)
               for index, stamp in enumerate(times)):
            raise ValueError('showinfo frame PTS cadence is not the requested 24fps')
        count, rate = len(events), geometry['fps']
        method = 'full input ffmpeg streaminfo plus -xerror RGB all-frame byte count and showinfo PTS/size validation'
    if count != geometry['num_frames'] or dimensions != (geometry['height'], geometry['width']) or not math.isclose(rate, geometry['fps'], rel_tol=0, abs_tol=1e-4):
        raise ValueError(f'Decoded media mismatch: frames={count}, dimensions={dimensions}, fps={rate}')
    return dict(verified=True, verification_method=method, frames=count, height=dimensions[0], width=dimensions[1],
                fps=rate, video_streams=1, audio_streams=0, complete_decode=True)


def load_lora(pipe, path):
    path = Path(path)
    if not path.is_file() or path.suffix != '.safetensors':
        raise ValueError('--lora requires the native already-scaled ModelLogger safetensors export, not a resume state')
    from safetensors import safe_open
    with safe_open(str(path), framework='pt', device='cpu') as stream:
        keys = list(stream.keys())
    pattern = re.compile(r'^(?:token_refiner\.)?blocks\.\d+\.(?:attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2))\.lora_[AB]\.default\.weight$')
    if not keys or any(not pattern.fullmatch(key) for key in keys):
        raise ValueError('Unexpected native H3 attention/MLP LoRA keys; no guessed prefix conversion')
    if any((key.replace('.lora_A.', '.lora_B.') if '.lora_A.' in key else key.replace('.lora_B.', '.lora_A.')) not in keys for key in keys):
        raise ValueError('Incomplete LoRA A/B pairs')
    record = source(path)
    pipe.load_lora(pipe.dit, lora_config=str(path), alpha=1.)
    return dict(file=record, keys=keys, alpha=1., scale_policy='B already contains training alpha/r; no second multiplication')


def run(args):
    import torch
    from h3_video_only import build_video_only_pipeline
    from h3_video_only import video_prompt
    out = check_new(args.out)
    if args.cache_only and (args.condition_cache or args.lora):
        raise ValueError('--cache-only cannot consume a condition cache or DiT LoRA')
    records = captions(args.caption_json, args.limit)
    geometry = dict(height=HEIGHT, width=WIDTH, num_frames=FRAMES, fps=FPS)
    if args.cfg != 1. or args.steps != 50 or args.seed != 42:
        raise ValueError('This experiment fixes 50 steps, seed42 and distilled CFG1')
    caption_source, code_source = source(args.caption_json), source(__file__)
    wrapper_source = source(Path(__file__).parent / 'h3_video_only.py')
    inputs = {row['id']: dict(caption_json=caption_source, image=source(row['image_path'])) for row in records}
    cached = bool(args.condition_cache)
    if not args.cache_only and not cached:
        raise ValueError('Use two isolated stages: --cache-only first, then --condition-cache for DiT+VAE generation')
    models = weight_sources(args.weights_root, args.cache_only, cached)
    pipe = build_video_only_pipeline(args.weights_root, device='cuda', torch_dtype=torch.bfloat16,
                                      load_text_encoder=not cached, load_mode='cache' if args.cache_only else 'all',
                                      vram_management=True)
    if pipe.audio_vae is not None or pipe.scheduler_audio is not None:
        raise AssertionError('Unexpected audio modules after model loading')
    lora_source = load_lora(pipe, args.lora) if args.lora else None
    if cached and pipe.text_encoder is not None:
        raise AssertionError('External cached generation unexpectedly loaded the TC')
    guard = AudioGuard(pipe)
    out.mkdir(exist_ok=False)
    run_audit = dict(protocol=PROTOCOL, source_code=code_source, wrapper_source=wrapper_source,
                     caption_source=caption_source, model_sources=models, lora_source=lora_source,
                     mode='condition_cache_only' if args.cache_only else 'video_generation',
                     geometry=geometry, steps=args.steps, seed=args.seed, cfg=1., flow_shift=args.flow_shift,
                     vram_management=True, old_y_reused=False, complete=False, completed_ids=[])
    atomic_json(out / 'run_started.json', run_audit)
    try:
        for row in records:
            condition_source = None
            if cached:
                cache_path = Path(args.condition_cache) / (row['id'] + '.condition.pt')
                condition_source = source(cache_path)
                cache_report = read_json(Path(args.condition_cache) / 'run_complete.json')
                case_report = read_json(Path(args.condition_cache) / (row['id'] + '.audit.json'))
                if cache_report.get('complete') is not True or row['id'] not in cache_report['completed_ids'] or case_report.get('complete') is not True or case_report['condition_cache']['sha256'] != condition_source['sha256']:
                    raise ValueError('Condition cache needs complete run/case audits and exact cache SHA; partial cache forbidden')
                condition, envelope = load_condition(cache_path, row, inputs[row['id']], geometry, args.seed, pipe.device)
                available = {str(Path(file['path']).relative_to(args.weights_root)): file['sha256'] for file in models['files']}
                for file in envelope['model_sources']['files']:
                    relative = str(Path(file['path']).relative_to(envelope['model_sources']['weights_root']))
                    if relative.startswith('video_vae/') and available.get(relative) != file['sha256']:
                        raise ValueError(f'Cached videoVAE weight version mismatch: {relative}')
                packing = packed_audit(condition['packed'])
            else:
                with torch.inference_mode():
                    condition, prompt, packing = prepare_condition(pipe, row, geometry, args.seed)
                envelope = cache_envelope(row, condition, prompt, geometry, args.seed, inputs[row['id']], models)
                cache_path = out / (row['id'] + '.condition.pt')
                atomic_torch(cache_path, envelope)
                condition_source = source(cache_path)
            result = dict(protocol=PROTOCOL, id=row['id'], caption_version=row['caption_version'],
                          motion_caption=row['motion_caption'], prompt=video_prompt(row['motion_caption']),
                          sources=inputs[row['id']], condition_cache=condition_source,
                          packed=packing, mode=run_audit['mode'], complete=False)
            if not args.cache_only:
                frames, sampling = sample_cached(pipe, condition, args.steps, args.seed, args.flow_shift)
                video_path = out / (row['id'] + '.mp4')
                media = write_video(video_path, frames, geometry)
                result.update(video=source(video_path), media=media, sampling=sampling)
                del frames
            result['audio_guard'] = guard.report()
            result['complete'] = True
            if cached and result['audio_guard']['observed_video_calls']['text_encoder']:
                raise AssertionError('Cached generation executed a text encoder')
            if 'review_music_sidecar' in row:
                result['review_music_sidecar'] = row['review_music_sidecar']
                result['review_music_consumed_by_h3'] = False
            atomic_json(out / (row['id'] + '.audit.json'), result)
            run_audit['completed_ids'].append(row['id'])
            print(json.dumps(dict(id=row['id'], complete=True, mode=run_audit['mode'], audio_guard=guard.report())), flush=True)
        for item in (caption_source, code_source, wrapper_source):
            if source(item['path'])['sha256'] != item['sha256']:
                raise RuntimeError('Source changed during run: ' + item['path'])
        run_audit.update(complete=True, audio_guard=guard.report())
        atomic_json(out / 'run_complete.json', run_audit)
    finally:
        guard.close()


class CpuChecks(unittest.TestCase):
    def test_atomic_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'audit.json'
            atomic_json(path, {'first': True})
            with self.assertRaises(FileExistsError):
                atomic_json(path, {'bad': True})
            self.assertEqual(read_json(path), {'first': True})

    def test_caption_schema_rejects_old_av(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'image.png'
            from PIL import Image
            Image.new('RGB', (32, 32)).save(image)
            base = dict(id='fixture', image_path=str(image), motion_caption='The dancer extends the arms.', caption_version='fixture-v1')
            path = root / 'caption.json'
            atomic_json(path, [base])
            self.assertEqual(captions(path, 1)[0]['id'], 'fixture')
            bad = root / 'bad.json'
            atomic_json(bad, [dict(base, motion_caption='<Audio 1> reused as soundtrack')])
            with self.assertRaises(ValueError):
                captions(bad, 1)

    def test_audio_guard_and_packed_real_tensors(self):
        import torch
        from types import SimpleNamespace
        dit = torch.nn.Module()
        dit.audio_patch_proj = torch.nn.Linear(2, 2)
        dit.audio_out = torch.nn.Linear(2, 2)
        pipe = SimpleNamespace(audio_vae=None, scheduler_audio=None, dit=dit, text_encoder=None, video_vae=None, units=[])
        guard = AudioGuard(pipe)
        self.assertEqual(guard.report()['calls']['audio_patch'], 0)
        with self.assertRaises(AssertionError):
            dit.audio_patch_proj(torch.zeros(1, 2))
        guard.close()
        good = dict(audio_pos=torch.empty(0, dtype=torch.long), token_tags=torch.tensor([0, 1]),
                    img_pos=torch.tensor([0]), text_pos=torch.tensor([1]))
        self.assertEqual(packed_audit(good)['tag2_count'], 0)
        with self.assertRaises(AssertionError):
            packed_audit(dict(good, token_tags=torch.tensor([0, 2])))

    def test_new_condition_cache_roundtrip_and_old_y_rejection(self):
        import torch
        from h3_video_only import video_prompt
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / 'image.png'
            from PIL import Image
            Image.new('RGB', (32, 32)).save(image_path)
            caption_path = root / 'caption.json'
            record = dict(id='fixture', image_path=str(image_path), motion_caption='The dancer extends the arms.', caption_version='fixture-v1')
            atomic_json(caption_path, record)
            inputs = dict(image=source(image_path), caption_json=source(caption_path))
            geometry = dict(height=32, width=32, num_frames=5, fps=24)
            condition = dict(prompt_embeds=torch.randn(3, 5120), text_token_tags=torch.ones(3, dtype=torch.long),
                             packed=dict(audio_pos=torch.empty(0, dtype=torch.long), token_tags=torch.tensor([0, 1]),
                                         img_pos=torch.tensor([0]), text_pos=torch.tensor([1])),
                             ref_visual_anchor=torch.randn(1, 96), ref_blocks=[], video_latent_shape=[1, 24, 2, 2, 2])
            envelope = cache_envelope(record, condition, video_prompt(record['motion_caption']), geometry, 42, inputs, {'files': []})
            path = root / 'fixture.condition.pt'
            atomic_torch(path, envelope)
            restored, metadata = load_condition(path, record, inputs, geometry, 42, 'cpu')
            self.assertTrue(torch.equal(restored['prompt_embeds'], condition['prompt_embeds']))
            self.assertTrue(torch.equal(restored['text_token_tags'], condition['text_token_tags']))
            self.assertFalse(metadata['old_y_reused'])
            with self.assertRaises(ValueError):
                load_condition(path, dict(record, caption_version='wrong'), inputs, geometry, 42, 'cpu')
            old = root / 'old.pt'
            atomic_torch(old, {'format': 'old_y'})
            with self.assertRaises(ValueError):
                load_condition(old, record, inputs, geometry, 42, 'cpu')

    def test_real_video_only_tiny_H3_sampling_counter(self):
        import torch
        from h3_video_only import H3VideoOnlyPipeline, MiniMaxH3DiT, enable_video_only_dit, VideoOnlyPacked
        pipe = H3VideoOnlyPipeline(torch_dtype=torch.float32)
        model = MiniMaxH3DiT(num_layers=2, token_refiner_num_layers=1, hidden_size=48,
                            num_attention_heads=4, attention_head_dim=12, ffn_hidden_size=64,
                            time_embed_dim=32, time_embed_hidden_size=48, adaln_out_features=864,
                            final_adaln_out_features=96, rope_inv_freq_len=2).float()
        enable_video_only_dit(model)
        pipe.dit = model
        class TestVideoVAE(torch.nn.Module):
            def decode_video(self, latents, **kwargs):
                return torch.zeros(1, 3, 5, 32, 32)
        pipe.video_vae = TestVideoVAE()
        shape = [1, 24, 2, 4, 4]
        text = torch.randn(3, 5120)
        tags = torch.ones(3, dtype=torch.long)
        packed = VideoOnlyPacked().process(pipe, prompt_embeds=text, video_latents=torch.zeros(shape),
                                            audio_latents=torch.empty(2, 32, 0), text_token_tags=tags)['packed']
        condition = dict(video_latent_shape=shape, prompt_embeds=text, text_token_tags=tags,
                         packed=packed, ref_visual_anchor=None, ref_blocks=None)
        guard = AudioGuard(pipe)
        frames, report = sample_cached(pipe, condition, 2, 42, 12.)
        self.assertEqual(report['video_scheduler_updates'], 2)
        self.assertEqual(len(frames), 5)
        self.assertEqual(guard.report()['calls']['audio_head'], 0)
        self.assertEqual(guard.report()['observed_video_calls']['dit'], 2)
        self.assertEqual(guard.report()['observed_video_calls']['video_vae_decode'], 1)
        self.assertEqual(guard.report()['observed_video_calls']['text_encoder'], 0)
        guard.close()

    def test_full_native_condition_PIL_cache_reload_sampling_and_scaled_lora(self):
        import torch
        from types import SimpleNamespace
        from h3_video_only import H3VideoOnlyPipeline, MiniMaxH3DiT, enable_video_only_dit
        from PIL import Image
        from safetensors.torch import save_file
        pipe = H3VideoOnlyPipeline(torch_dtype=torch.float32)
        pipe.dit = MiniMaxH3DiT(num_layers=2, token_refiner_num_layers=1, hidden_size=48,
                              num_attention_heads=4, attention_head_dim=12, ffn_hidden_size=64,
                              time_embed_dim=32, time_embed_hidden_size=48, adaln_out_features=864,
                              final_adaln_out_features=96, rope_inv_freq_len=2).float()
        enable_video_only_dit(pipe.dit)
        class TinyTokenizer:
            def __call__(self, text, **kwargs):
                return {'input_ids': [1] * max(1, len(text.split()))}
            def convert_tokens_to_ids(self, text):
                return 2
        class TinyImageProcessor:
            merge_size = 2
            def __call__(self, images, **kwargs):
                return {'pixel_values': torch.zeros(4, 12), 'image_grid_thw': torch.tensor([[1, 2, 2]])}
        class TinyTC(torch.nn.Module):
            def forward(self, input_ids, pixel_values=None, **kwargs):
                if pixel_values is None:
                    raise AssertionError('Native Ref2VA must pass visual tokens to TC')
                return torch.zeros(input_ids.shape[1], 5120)
        class TinyVAE(torch.nn.Module):
            def encode_video(self, image, **kwargs):
                if kwargs.get('process_image') is not True:
                    raise AssertionError('Image reference VAE path expected')
                return torch.zeros(1, 24, 1, 4, 4)
            def decode_video(self, latents, **kwargs):
                return torch.zeros(1, 3, 5, 32, 32)
        pipe.tokenizer = TinyTokenizer()
        pipe.processor = SimpleNamespace(image_processor=TinyImageProcessor())
        pipe.text_encoder, pipe.video_vae = TinyTC(), TinyVAE()
        geometry = dict(height=32, width=32, num_frames=5, fps=24)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / 'reference.png'
            Image.new('RGB', (32, 32), 'gray').save(image_path)
            record = dict(id='native-fixture', image_path=str(image_path), motion_caption='The dancer bends and extends the arms.', caption_version='CPU-fixture-v1')
            caption_path = root / 'captions.json'
            atomic_json(caption_path, record)
            guard = AudioGuard(pipe)
            condition, prompt, report = prepare_condition(pipe, record, geometry, 42)
            self.assertEqual(guard.report()['observed_video_calls']['text_encoder'], 1)
            self.assertEqual(guard.report()['observed_video_calls']['video_vae_encode'], 1)
            self.assertEqual(report['tag2_count'], 0)
            self.assertTrue(condition['ref_blocks'][0]['prepared_image_rgb'].dtype == torch.uint8)
            inputs = dict(image=source(image_path), caption_json=source(caption_path))
            envelope = cache_envelope(record, condition, prompt, geometry, 42, inputs, {'files': []})
            cache_path = root / 'native.condition.pt'
            atomic_torch(cache_path, envelope)
            restored, _ = load_condition(cache_path, record, inputs, geometry, 42, 'cpu')
            guard.close()
            pipe.text_encoder = pipe.processor = pipe.tokenizer = None
            guard = AudioGuard(pipe)
            frames, report = sample_cached(pipe, restored, 2, 42, 12.)
            self.assertEqual(len(frames), 5)
            self.assertEqual(guard.report()['observed_video_calls']['text_encoder'], 0)
            self.assertEqual(guard.report()['observed_video_calls']['dit'], 2)
            guard.close()
            import copy
            expected_dit = copy.deepcopy(pipe.dit)
            state = {}
            for prefix in ('blocks.0.attn.out_proj', 'token_refiner.blocks.0.attn.out_proj'):
                target = pipe.dit.get_submodule(prefix)
                original = target.weight.detach().clone()
                a = torch.randn(2, original.shape[1]) * .01
                b = torch.randn(original.shape[0], 2) * .01 * 2  # Exported B already includes alpha/r.
                state[prefix + '.lora_A.default.weight'] = a
                state[prefix + '.lora_B.default.weight'] = b
                expected_dit.get_submodule(prefix).weight.data.copy_(original + b @ a)
            lora = root / 'native_scaled.safetensors'
            save_file(state, str(lora))
            loading = load_lora(pipe, lora)
            self.assertEqual(loading['alpha'], 1.)
            for prefix in ('blocks.0.attn.out_proj', 'token_refiner.blocks.0.attn.out_proj'):
                self.assertTrue(torch.allclose(pipe.dit.get_submodule(prefix).weight,
                                               expected_dit.get_submodule(prefix).weight, atol=1e-7, rtol=1e-5))
            video = torch.randn(restored['video_latent_shape'])
            values = {key: restored[key] for key in ('prompt_embeds', 'packed', 'ref_visual_anchor', 'ref_blocks')}
            actual, empty = pipe.model_fn(dit=pipe.dit, video_latents=video, timestep_video=torch.tensor([500.]), **values)
            expected, _ = pipe.model_fn(dit=expected_dit, video_latents=video, timestep_video=torch.tensor([500.]), **values)
            self.assertEqual(empty.numel(), 0)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))
            for missing in ('blocks.0.attn.out_proj.lora_A.default.weight',
                            'token_refiner.blocks.0.attn.out_proj.lora_B.default.weight'):
                incomplete = root / ('incomplete_' + ('A' if '.lora_A.' in missing else 'B') + '.safetensors')
                save_file({key: value for key, value in state.items() if key != missing}, str(incomplete))
                with self.assertRaises(ValueError):
                    load_lora(pipe, incomplete)
            guard = AudioGuard(pipe)
            _, report = sample_cached(pipe, restored, 2, 42, 12.)
            self.assertEqual(guard.report()['calls']['audio_patch'], 0)
            self.assertEqual(guard.report()['calls']['audio_head'], 0)
            guard.close()

    def test_actual_portrait_media_all_frames_no_audio(self):
        import numpy as np
        geometry = dict(height=HEIGHT, width=WIDTH, num_frames=FRAMES, fps=FPS)
        with tempfile.TemporaryDirectory() as directory:
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            path = Path(directory) / 'fixture.mp4'
            result = write_video(path, (frame for _ in range(FRAMES)), geometry)
            self.assertEqual(result['frames'], 124)
            self.assertEqual(result['audio_streams'], 0)
            self.assertTrue(result['complete_decode'])
            with self.assertRaises(ValueError):
                verify_media(path, dict(geometry, num_frames=123))
            with self.assertRaises(ValueError):
                verify_media(path, dict(geometry, fps=12))
            with self.assertRaises(ValueError):
                verify_media(path, dict(geometry, width=672))
            with self.assertRaises(FileExistsError):
                write_video(path, [frame], geometry)
            av = Path(directory) / 'fixture_audio.mp4'
            subprocess.run([ffmpeg_binary(), '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
                            '-i', str(path), '-f', 'lavfi', '-i', 'anullsrc=r=24000:cl=mono',
                            '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(av)], check=True)
            with self.assertRaises(ValueError):
                verify_media(av, geometry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--caption-json', type=Path)
    parser.add_argument('--weights-root', type=Path, default=Path('/dev/shm/h3/Ref2VA'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--cache-only', action='store_true')
    parser.add_argument('--condition-cache', type=Path, help='Directory of this entrypoint new .condition.pt files; no TC loaded')
    parser.add_argument('--lora', type=Path, help='Native already-scaled H3 ModelLogger safetensors export; alpha fixed at 1')
    parser.add_argument('--limit', type=int, default=1, help='Early one case; explicitly 3 for fixed evaluation cases')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cfg', type=float, default=1.)
    parser.add_argument('--flow-shift', type=float, default=12.)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CpuChecks))
        if not result.wasSuccessful():
            raise SystemExit(1)
    else:
        if args.caption_json is None or args.out is None:
            parser.error('--caption-json and --out are required')
        run(args)


if __name__ == '__main__':
    main()
