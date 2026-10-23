"""Music-only Thinker SFT. selftest is a CPU fixture, not a pretrained-model smoke.

Metadata JSONL: id, source_id/source_group, split (train/val), audios [path],
messages [user '<audio>instruction', assistant caption] (optional initial system).
Optional audio_window_seconds [start, end] is applied before 16 kHz resampling.
Use train --smoke-steps 2 for two real optimizer updates, not a completed SFT.
"""
import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict


SCHEMA = 'omni-dance-sft-v1'
PLACEHOLDER = '<audio>'
SOURCE_LABEL_PROTOCOL = 'omni-source-label-pilot-v1'
SOURCE_LABEL_AUTHORIZATION = 'User authorized simple MotionScript + style pilot on 2026-10-23'


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def log(value, rank=0):
    if rank == 0:
        print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def text_content(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and all(p.get('type') == 'text' for p in value):
        return ''.join(p['text'] for p in value)
    raise ValueError('System/assistant content must contain text only')


def conversation(row, require_caption=True):
    messages = row['messages']
    roles = [m['role'] for m in messages]
    if not require_caption and roles in (['user', 'assistant'], ['system', 'user', 'assistant']):
        messages = messages[:-1]
        roles = roles[:-1]
    expected = (['user', 'assistant'], ['system', 'user', 'assistant']) if require_caption else (['user'], ['system', 'user'])
    if roles not in expected:
        raise ValueError('Require optional system, exactly one user, and one assistant only for training')
    audios = row['audios']
    if not isinstance(audios, list) or len(audios) != 1 or not isinstance(audios[0], str):
        raise ValueError('Music-only v1 requires exactly one audios path')
    user = messages[-2 if require_caption else -1]['content']
    if isinstance(user, str):
        if user.count(PLACEHOLDER) != 1:
            raise ValueError('String user content must contain exactly one <audio>')
        before, after = user.split(PLACEHOLDER)
        parts = ([dict(type='text', text=before)] if before else [])
        parts += [dict(type='audio', audio=audios[0])]
        parts += [dict(type='text', text=after)] if after else []
    elif isinstance(user, list):
        if any(p.get('type') not in ('text', 'audio') for p in user):
            raise ValueError('Images/video are forbidden in music-only SFT')
        if sum(p.get('type') == 'audio' for p in user) != 1:
            raise ValueError('Require one audio content part')
        parts = []
        for part in user:
            if part['type'] == 'audio':
                supplied = part.get('audio', part.get('audio_url'))
                if supplied is not None and supplied != audios[0]:
                    raise ValueError('Audio placeholder and audios path disagree')
                parts.append(dict(type='audio', audio=audios[0]))
            else:
                parts.append(dict(type='text', text=part['text']))
    else:
        raise ValueError('Invalid user content')
    caption = text_content(messages[-1]['content']) if require_caption else None
    if require_caption and (not caption.strip() or any(x in caption for x in ('<|', PLACEHOLDER))):
        raise ValueError('Empty caption or reserved token in assistant caption')
    for part in parts:
        if part['type'] == 'text' and ('<|' in part['text'] or PLACEHOLDER in part['text']):
            raise ValueError('Reserved multimodal/chat tokens in instruction')
        if part['type'] == 'text' and caption is not None and caption.strip() in part['text']:
            raise ValueError('Assistant caption leaked into user instruction')
    result = []
    if roles[0] == 'system':
        system = text_content(messages[0]['content'])
        if '<|' in system or PLACEHOLDER in system or caption is not None and caption.strip() in system:
            raise ValueError('Reserved token or caption leakage in system')
        result.append(dict(role='system', content=[dict(type='text', text=system)]))
    result.append(dict(role='user', content=parts))
    if require_caption:
        result.append(dict(role='assistant', content=[dict(type='text', text=caption)]))
    return result


def read_metadata(path, training=True):
    rows = []
    ids = set()
    groups = {'train': set(), 'val': set()}
    audio_groups = {'train': set(), 'val': set()}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('split') not in groups or not isinstance(row.get('id'), str):
            raise ValueError(f'Line {number}: require string id and explicit train/val split')
        if row['id'] in ids:
            raise ValueError(f'Duplicate id: {row["id"]}')
        ids.add(row['id'])
        group = row.get('source_group', row.get('source_id'))
        if not isinstance(group, str) or not group:
            raise ValueError(f'Line {number}: missing source isolation group')
        groups[row['split']].add(group)
        conversation(row, require_caption=training)
        audio = Path(row['audios'][0]).expanduser()
        audio = audio if audio.is_absolute() else Path(path).resolve().parent / audio
        row = copy.deepcopy(row)
        # Resolve a structured placeholder before it is validated a second time.
        for message in row['messages']:
            if isinstance(message['content'], list):
                for part in message['content']:
                    if part.get('type') == 'audio':
                        part.pop('audio_url', None)
                        part['audio'] = str(audio.resolve())
        row['audios'] = [str(audio.resolve())]
        if not audio.is_file():
            raise ValueError(f'Audio missing: {audio}')
        audio_groups[row['split']].add(str(audio.resolve()))
        window = row.get('audio_window_seconds')
        if window is not None and (not isinstance(window, list) or len(window) != 2 or
                                   any(type(v) not in (float, int) or not math.isfinite(v) for v in window) or
                                   not 0 <= window[0] < window[1]):
            raise ValueError(f'Invalid explicit audio window: {row["id"]}')
        rows.append(row)
    if groups['train'] & groups['val'] or audio_groups['train'] & audio_groups['val']:
        raise ValueError('Train/val source group or audio file leakage')
    train = [r for r in rows if r['split'] == 'train']
    val = [r for r in rows if r['split'] == 'val']
    if training and (not train or not val):
        raise ValueError('Metadata must contain nonempty fixed train and val splits')
    return rows, train, val


def read_audio(row):
    import soundfile as sf
    from scipy.signal import resample_poly
    with sf.SoundFile(row['audios'][0]) as source:
        sr = source.samplerate
        window = row.get('audio_window_seconds')
        if window is None:
            start, end = 0, source.frames
        else:
            start, end = [round(float(s) * sr) for s in window]
        if not 0 <= start < end <= source.frames:
            raise ValueError(f'Audio window outside source: {row["id"]}')
        source.seek(start)
        waveform = source.read(end - start, dtype='float32', always_2d=True).mean(axis=1)
    if sr != 16000:
        divisor = math.gcd(sr, 16000)
        waveform = resample_poly(waveform, 16000 // divisor, sr // divisor).astype('float32')
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError(f'Invalid audio: {row["id"]}')
    return waveform


def mask_response(full, prefix, eos_id):
    ids = full['input_ids']
    prefix_ids = prefix['input_ids'][0][prefix['attention_mask'][0].bool()]
    valid = full['attention_mask'][0].bool()
    indices = valid.nonzero().flatten()
    tokens = ids[0, valid]
    n = len(prefix_ids)
    if n >= len(tokens) or not torch.equal(tokens[:n], prefix_ids):
        raise ValueError('Chat prefix is not an exact processed token prefix; refusing approximate label boundary')
    eos_positions = (tokens[n:] == eos_id).nonzero().flatten()
    if len(eos_positions) != 1:
        raise ValueError('Require exactly one assistant EOS after processed prefix')
    stop = n + int(eos_positions[0]) + 1
    labels = torch.full_like(ids, -100)
    labels[0, indices[n:stop]] = tokens[n:stop]
    if int((labels[:, 1:] != -100).sum()) < 2 or labels[0, indices[stop - 1]] != eos_id:
        raise ValueError('Assistant caption/EOS supervision missing')
    return labels


def prepare_inputs(processor, row, training=True, max_length=32768, waveform=None, audio_n_window=None):
    """Shared training/deployment boundary; processor expands the real audio tokens."""
    messages = conversation(row, require_caption=training)
    waveform = read_audio(row) if waveform is None else waveform
    audio_kwargs = dict(sampling_rate=16000, return_attention_mask=True, truncation=False)
    if audio_n_window is not None:
        audio_kwargs['n_window'] = audio_n_window
    kwargs = dict(audio=[waveform], return_tensors='pt', padding=False,
                  text_kwargs=dict(add_special_tokens=False), audio_kwargs=audio_kwargs,
                  videos_kwargs=dict(use_audio_in_video=False))
    prefix_text = processor.apply_chat_template(messages[:-1] if training else messages,
                                                 add_generation_prompt=True, tokenize=False)
    prefix = processor(text=prefix_text, **kwargs)
    if training:
        full_text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        full = processor(text=full_text, **kwargs)
        full['labels'] = mask_response(full, prefix, processor.tokenizer.eos_token_id)
        for key in ('input_features', 'feature_attention_mask'):
            if not torch.equal(prefix[key], full[key]):
                raise ValueError('Audio features changed between training and deployment prefix')
    else:
        full = prefix
    if full['input_ids'].shape[-1] > max_length:
        raise ValueError(f'Context exceeds limit for {row["id"]}; truncation is forbidden')
    audio_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_token)
    if int((full['input_ids'] == audio_id).sum()) <= 0 or 'input_features' not in full:
        raise ValueError('Real audio features/expanded audio placeholders missing')
    if any(k.startswith('pixel_values') or 'grid_thw' in k for k in full):
        raise ValueError('Unexpected visual features in music-only input')
    if training and torch.any(full['labels'][full['input_ids'] == audio_id] != -100):
        raise ValueError('Audio placeholder was supervised')
    return full


def model_classes(family):
    if family == 'qwen2.5':
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration
        return Qwen2_5OmniProcessor, Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration
    from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerConfig, Qwen3OmniMoeThinkerForConditionalGeneration
    return Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerConfig, Qwen3OmniMoeThinkerForConditionalGeneration


def identify_family(model_path):
    cfg = json.loads((Path(model_path) / 'config.json').read_text())
    kind = cfg['model_type']
    if kind.startswith('qwen2_5_omni'):
        return 'qwen2.5'
    if kind.startswith('qwen3_omni_moe'):
        return 'qwen3'
    raise ValueError(f'Unsupported model_type: {kind}')


def attach_lora(model, family, rank=32, alpha=64):
    model.requires_grad_(False)
    audio_layers = len(model.audio_tower.layers)
    if audio_layers < 4:
        raise ValueError('Audio tower must contain at least four Transformer blocks')
    selected = []
    expert_parameters = []
    for name, module in model.named_modules():
        is_lm = re.fullmatch(r'model\.layers\.\d+\.(self_attn|mlp)(\..+)?', name)
        is_audio = re.fullmatch(r'audio_tower\.layers\.(\d+)\..+', name)
        if isinstance(module, torch.nn.Linear) and (is_lm or (is_audio and int(is_audio[1]) >= audio_layers - 4)):
            selected.append(name)
    for name, parameter in model.named_parameters():
        if re.fullmatch(r'model\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)', name):
            if parameter.ndim != 3:
                raise ValueError(f'Unexpected fused expert shape: {name}')
            expert_parameters.append(name)
    projectors = ['audio_tower.proj'] if family == 'qwen2.5' else ['audio_tower.proj1', 'audio_tower.proj2']
    for name in projectors:
        if not isinstance(model.get_submodule(name), torch.nn.Linear):
            raise ValueError(f'Unexpected audio projector: {name}')
    if (not any(n.startswith('model.layers.') and '.self_attn.' in n for n in selected) or
            not (any(n.startswith('model.layers.') and '.mlp.' in n for n in selected) or expert_parameters)):
        raise ValueError('No LLM attention/MLP targets')
    if family == 'qwen3' and model.config.text_config.num_experts > 0 and not expert_parameters:
        raise ValueError('Qwen3 MoE expert MLP would remain frozen')
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0., bias='none',
                        target_modules=selected, target_parameters=expert_parameters or None,
                        modules_to_save=projectors)
    model = get_peft_model(model, config)
    scope = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if not ('lora_' in name or '.modules_to_save.' in name):
                raise ValueError(f'Unexpected trainable base parameter: {name}')
            if any(s in name.lower() for s in ('visual', 'talker', 'code2wav', 'lm_head')):
                raise ValueError(f'Forbidden trainable parameter: {name}')
            scope.append(dict(name=name, shape=list(parameter.shape), parameters=parameter.numel(),
                              group='projector' if '.modules_to_save.' in name else 'lora'))
    if hasattr(model.get_base_model(), 'talker') or hasattr(model.get_base_model(), 'code2wav'):
        raise ValueError('Thinker loader unexpectedly created speech-generation modules')
    return model, dict(target_modules=selected, target_parameters=expert_parameters,
                       projectors=projectors, audio_last_four=list(range(audio_layers - 4, audio_layers)),
                       trainable=scope)


def forbid_visual(model):
    def reject(*args, **kwargs):
        raise RuntimeError('Visual tower called in music-only mode')
    return model.get_base_model().visual.register_forward_pre_hook(reject)


def make_optimizer(model, lora_lr, projector_lr, zero_stage1=False):
    groups = [dict(params=[], lr=lora_lr, name='lora'), dict(params=[], lr=projector_lr, name='projector')]
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups[int('.modules_to_save.' in name)]['params'].append(parameter)
    if not all(g['params'] for g in groups):
        raise ValueError('Both LoRA and audio projector optimizer groups are required')
    if zero_stage1:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        return ZeroRedundancyOptimizer(groups, optimizer_class=torch.optim.AdamW, weight_decay=0.)
    return torch.optim.AdamW(groups, weight_decay=0.)


def to_device(inputs, device, dtype):
    return {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype) for k, v in inputs.items()}


def shifted_nll(logits, labels):
    count = (labels[:, 1:] != -100).sum()
    nll = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                          labels[:, 1:].reshape(-1), ignore_index=-100, reduction='sum')
    return nll, count


def gradient_report(model):
    report = []
    categories = dict(lm=0, audio_lora=0, projector=0)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise ValueError(f'Frozen parameter has a gradient: {name}')
            continue
        gradient = parameter.grad
        finite = gradient is None or bool(torch.isfinite(gradient).all())
        if not finite:
            raise FloatingPointError(f'Nonfinite gradient: {name}')
        norm = float(gradient.float().norm()) if gradient is not None else None
        report.append(dict(name=name, gradient_norm=norm))
        category = 'projector' if '.modules_to_save.' in name else 'audio_lora' if 'audio_tower.' in name else 'lm'
        categories[category] += int(norm is not None and norm > 0)
    if not all(categories.values()):
        raise ValueError(f'Missing real gradient in required scope: {categories}')
    return dict(parameters=report, nonzero_by_scope=categories)


def rng_state(device):
    value = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    if device.type == 'cuda':
        value['cuda'] = torch.cuda.get_rng_state(device)
    return value


def restore_rng(value, device):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(value['cuda'], device)


def tensor_state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(out, model, optimizer, state, fingerprint, device, rank, world, zero_stage1=False):
    if zero_stage1:
        optimizer.consolidate_state_dict(to=0)
    states = [None] * world if rank == 0 else None
    local_rng = rng_state(device)
    if world > 1:
        dist.gather_object(local_rng, states, dst=0)
    else:
        states = [local_rng]
    checkpoint_name = f'checkpoint-u{state["updates"]:07d}-e{state["epoch"]:02d}-p{state["position"]:07d}'
    def commit_checkpoint():
        checkpoint = out / checkpoint_name
        if checkpoint.exists():
            raise FileExistsError(f'Checkpoint already exists: {checkpoint}')
        temporary = Path(tempfile.mkdtemp(prefix='.checkpoint-', dir=out))
        model.save_pretrained(temporary / 'adapter', safe_serialization=True, save_embedding_layers=False)
        saved_state = dict(schema=SCHEMA, fingerprint=fingerprint, optimizer=optimizer.state_dict(),
                           rng_by_rank=states, sampler=copy.deepcopy(state), world_size=world)
        torch.save(saved_state, temporary / 'training_state.pt')
        hashes = {str(p.relative_to(temporary)): file_sha(p) for p in temporary.rglob('*') if p.is_file()}
        atomic_json(temporary / 'complete.json', dict(schema=SCHEMA, files=hashes, fingerprint=fingerprint,
                    updates=state['updates'], epoch=state['epoch'], position=state['position']))
        os.replace(temporary, checkpoint)
        atomic_json(out / 'latest.json', dict(checkpoint=checkpoint_name, updates=state['updates']))
    main_process_result(commit_checkpoint, rank, world)
    return out / checkpoint_name


def validate_cursor(state, train_size, max_epochs):
    required = {'epoch', 'position', 'permutation', 'updates', 'examples', 'tokens',
                'history', 'best_nll', 'stale_epochs', 'padded_forwards'}
    if set(state) != required:
        raise ValueError('Incomplete sampler/history resume state')
    if (type(state['epoch']) is not int or not 0 <= state['epoch'] <= max_epochs or
            type(state['position']) is not int or not 0 <= state['position'] <= train_size or
            any(type(state[k]) is not int or state[k] < 0 for k in ('updates', 'examples', 'tokens', 'stale_epochs', 'padded_forwards'))):
        raise ValueError('Invalid optimizer/sampler resume cursor')
    permutation = state['permutation']
    if permutation is None:
        if state['position'] != 0:
            raise ValueError('Resume position without sampler permutation')
    elif sorted(permutation) != list(range(train_size)):
        raise ValueError('Resume sampler permutation changed or incomplete')
    if len(state['history']) != state['epoch']:
        raise ValueError('Resume epoch/history mismatch')


def load_checkpoint(checkpoint, model, optimizer, fingerprint, device, rank, world):
    checkpoint = Path(checkpoint)
    complete = json.loads((checkpoint / 'complete.json').read_text())
    if complete['fingerprint'] != fingerprint or complete['schema'] != SCHEMA:
        raise ValueError('Checkpoint provenance/schema mismatch')
    for name, expected in complete['files'].items():
        if file_sha(checkpoint / name) != expected:
            raise ValueError(f'Incomplete/corrupt checkpoint file: {name}')
    state = torch.load(checkpoint / 'training_state.pt', map_location='cpu', weights_only=False)
    if state['fingerprint'] != fingerprint or state['world_size'] != world or state['schema'] != SCHEMA:
        raise ValueError('Full resume requires identical provenance, schema and world size')
    if any(state['sampler'][k] != complete[k] for k in ('updates', 'epoch', 'position')):
        raise ValueError('Checkpoint commit marker and resume cursor disagree')
    if len(state['rng_by_rank']) != world:
        raise ValueError('Checkpoint lacks complete per-rank RNG states')
    from safetensors.torch import load_file
    adapter = load_file(checkpoint / 'adapter' / 'adapter_model.safetensors')
    result = set_peft_model_state_dict(model, adapter)
    if result.unexpected_keys:
        raise ValueError(f'Unexpected adapter keys: {result.unexpected_keys}')
    if tensor_state_hash(get_peft_model_state_dict(model)) != tensor_state_hash(adapter):
        raise ValueError('Adapter/modules_to_save did not restore exactly')
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng_by_rank'][rank], device)
    return state['sampler']


def source_label_caption(record):
    import dance_caption_fusion as fusion
    sentences = record.get('source_MS', {}).get('ms')
    if not isinstance(sentences, list) or any(not isinstance(text, str) for text in sentences):
        raise ValueError('Source-label caption requires original MotionScript sentences')
    motion = ' '.join(text.strip() for text in sentences if text.strip())
    if not motion:
        raise ValueError('Source-label caption requires nonempty original MotionScript')
    motion = re.sub(r'\b(?:a few seconds (?:earlier|later)|in the second right (?:before|after)|(?:one|a) second (?:earlier|later))\b\s*[,;:]?\s*', '', motion, flags=re.I)
    motion = re.sub(r'\b(?:her|his)\b', "the dancer's", motion, flags=re.I)
    motion = re.sub(r'\b(?:she|he)\b', 'the dancer', motion, flags=re.I)
    motion = re.sub(r'\b(?:(?:the|this|a)\s+)?(?:subject|person|human)\b|\bsomeone\b', 'the dancer', motion, flags=re.I)
    motion = re.sub(r'\babout-face\b', 'halfway', motion, flags=re.I)
    motion = re.sub(r'\bface\s+to\b', 'oriented to', motion, flags=re.I)
    motion = re.sub(r'\s+', ' ', motion).strip()
    if not motion:
        raise ValueError('Source-label caption is empty after removing synthetic timing')
    for name, pattern in fusion.FORBIDDEN.items():
        if pattern.search(motion):
            raise ValueError(f'Unsafe original MotionScript boundary: {name}')
    motion = re.sub(r'^the dancer\b', 'The dancer', motion)
    if not motion.startswith('The dancer'):
        motion = 'The dancer: ' + motion
    genre = record.get('source_selected_caption', {}).get('dims', {}).get('genre')
    if not isinstance(genre, str) or not genre.strip() or re.search(r'[<>\n\r]', genre):
        raise ValueError('Source-label caption requires a plain original source genre')
    return f'Dance style: {genre}. {motion}'


def source_label_omni_provenance(metadata_path, caption_version, rows, training_rows, val_rows):
    """Verify source labels and media identity, explicitly without semantic approval."""
    import wave
    import dance_caption_fusion as fusion
    metadata_path = Path(metadata_path).resolve()
    sidecar_path = metadata_path.parent / 'source_label_sidecar.json'
    verified = {}
    def verify(source):
        if not isinstance(source, dict) or not isinstance(source.get('path'), str):
            raise ValueError('Missing source-label source record')
        actual = fusion.source_record(source['path'], source.get('sha256', 'missing'))
        if actual != {key: source.get(key) for key in ('path', 'sha256', 'size_bytes')}:
            raise ValueError('Source-label path/SHA/size differs from source record')
        verified[actual['path']] = actual
        return actual
    sidecar_source = fusion.source_record(sidecar_path)
    verified[sidecar_source['path']] = sidecar_source
    sidecar = fusion.read_json(sidecar_path)
    if (sidecar.get('protocol') != SOURCE_LABEL_PROTOCOL or sidecar.get('complete') is not True or
            sidecar.get('label_quality') != 'source_labels_unreviewed' or
            sidecar.get('authorization') != SOURCE_LABEL_AUTHORIZATION or
            sidecar.get('human_review_completed') is not False or
            sidecar.get('approval_chain_verified', False) is not False or
            sidecar.get('semantic_approval') is not False or
            sidecar.get('original_source_window_mapping') != 'unknown' or
            any(sidecar.get(key, False) is not False for key in ('fixture_only', 'formal_reuse_forbidden'))):
        raise ValueError('Source-label pilot requires explicit authorization and unreviewed unknown-window labels')
    builder_source = verify(sidecar['source_code'])
    manifest_source = verify(sidecar['manifest'])
    manifest = fusion.load_manifest(manifest_source['path'])
    if any(manifest.get(key, False) is not False for key in ('fixture_only', 'formal_reuse_forbidden', 'semantic_approval')):
        raise ValueError('Source-label manifest cannot claim semantic approval or fixture reuse')
    historical_producer = manifest['source_code']
    for key in ('caps', 'split'):
        if key in manifest['source_inputs']:
            verify(manifest['source_inputs'][key])
    selected_captions = fusion.read_json(manifest['source_inputs']['caps']['path'])
    if caption_version != SOURCE_LABEL_PROTOCOL + ':' + manifest_source['sha256']:
        raise ValueError('Source-label caption version differs from source manifest SHA')
    output = verify(sidecar['outputs']['omni.jsonl'])
    if metadata_path != Path(output['path']).resolve():
        raise ValueError('Source-label metadata must be the declared omni.jsonl')
    expected = []
    media_groups = {'audio': {'train': set(), 'val': set()}}
    for record in manifest['records']:
        sid = record['id']
        if (not isinstance(record.get('source_group'), str) or not record['source_group'] or
                any(record.get(key, False) is not False for key in ('fixture_only', 'formal_reuse_forbidden'))):
            raise ValueError('Invalid source-label record group or fixture identity')
        sources = record['provenance']
        for key in ('ms', 'audio'):
            verify(sources[key])
        source_ms = fusion.read_json(sources['ms']['path'])
        if source_ms != record['source_MS'] or source_ms.get('sid') != sid:
            raise ValueError('Source-label original MotionScript differs from its source or sid')
        if selected_captions.get(sid) != record['source_selected_caption']:
            raise ValueError('Source-label style differs from original selected caption source')
        if (record['audio_path'] != sources['audio']['path'] or
                record.get('audio_window_seconds') != [0., 5.0625]):
            raise ValueError('Source-label media paths or explicit audio window differ')
        with wave.open(sources['audio']['path'], 'rb') as audio:
            if (audio.getframerate() <= 0 or audio.getnchannels() not in (1, 2) or
                    audio.getnframes() / audio.getframerate() < 5.0625):
                raise ValueError('Source-label requested normalized audio window is unavailable')
        for key in media_groups:
            media_groups[key][record['split']].add(sources[key]['path'])
        expected.append(dict(id=sid, source_group=record['source_group'], split=record['split'],
            caption_version=caption_version,
            messages=[dict(role='user', content=fusion.SFT_INSTRUCTION + '\n<audio>'),
                      dict(role='assistant', content=source_label_caption(record))],
            audios=[sources['audio']['path']], audio_window_seconds=[0., 5.0625]))
    if any(groups['train'] & groups['val'] for groups in media_groups.values()):
        raise ValueError('Source-label normalized media leaks across train/val')
    raw_rows = [json.loads(line, object_pairs_hook=fusion.unique_object)
                for line in metadata_path.read_text().splitlines() if line.strip()]
    counts = dict(train=len(training_rows), val=len(val_rows))
    if (rows != expected or raw_rows != expected or
            training_rows != [row for row in expected if row['split'] == 'train'] or
            val_rows != [row for row in expected if row['split'] == 'val'] or
            sidecar.get('counts') != counts or manifest.get('counts') != counts or
            any(type(sidecar['counts'][key]) is not int for key in counts)):
        raise ValueError('Source-label exact Omni rows, partitions or source counts differ')
    for source in list(verified.values()):
        verify(source)
    return dict(protocol=SOURCE_LABEL_PROTOCOL, metadata=output, sidecar=sidecar_source,
        manifest=manifest_source, builder_source=builder_source,
        historical_manifest_producer=historical_producer,
        validator_source=fusion.source_record(fusion.__file__),
        all_verified_sources=list(verified.values()), caption_version=caption_version,
        authorization=SOURCE_LABEL_AUTHORIZATION, label_quality='source_labels_unreviewed',
        approval_chain_verified=False, human_review_completed=False, semantic_approval=False,
        original_source_window_mapping='unknown', fixture_only=False, formal_reuse_forbidden=False,
        train_ids=[row['id'] for row in training_rows], val_ids=[row['id'] for row in val_rows])


def formal_omni_provenance(metadata_path, caption_version, rows, training_rows, val_rows):
    """Reuse the complete H3 approval gate before comparing the exact Omni exports."""
    import dance_caption_fusion as fusion
    import h3_video_sft
    metadata_path = Path(metadata_path).resolve()
    sidecar_path = metadata_path.parent / 'source_sidecar.json'
    sidecar_source = fusion.source_record(sidecar_path)
    sidecar = fusion.read_json(sidecar_path)
    if metadata_path != Path(sidecar['outputs']['omni.jsonl']['path']).resolve():
        raise ValueError('Reviewed Omni training requires the declared omni.jsonl output')
    approval = h3_video_sft.formal_training_provenance(
        sidecar['outputs']['h3_video_train.json']['path'], caption_version)
    if fusion.source_record(sidecar_path) != sidecar_source or approval.get('approval_chain_verified') is not True:
        raise ValueError('Approval chain failed or the reviewed sidecar changed during validation')
    expected = []
    for record in sidecar['records']:
        if (not isinstance(record.get('id'), str) or not record['id'] or
                record.get('caption_version') != caption_version or
                record.get('split') not in ('train', 'val') or
                any(record.get(key, False) is not False for key in ('fixture_only', 'formal_reuse_forbidden'))):
            raise ValueError('Reviewed Omni record ID/version/split/fixture identity differs')
        expected.append(dict(id=record['id'], source_group=record['source_group'], split=record['split'],
            caption_version=caption_version,
            messages=[dict(role='user', content=fusion.SFT_INSTRUCTION + '\n<audio>'),
                      dict(role='assistant', content=record['motion_caption'])],
            audios=[record['audio_source']['path']], audio_window_seconds=record['audio_window_seconds']))
    def raw_jsonl(path):
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if rows != expected or raw_jsonl(metadata_path) != expected:
        raise ValueError('Omni metadata differs from the approved caption/source/window/ID/version/split export')
    for partition, actual in (('train', training_rows), ('val', val_rows)):
        selected = [row for row in expected if row['split'] == partition]
        if (actual != selected or raw_jsonl(sidecar['outputs'][partition + '.jsonl']['path']) != selected or
                type(sidecar.get('counts', {}).get(partition)) is not int or
                sidecar['counts'][partition] != len(selected)):
            raise ValueError('Reviewed Omni partition/counts differ from the approved export')
    return dict(protocol='omni-reviewed-training-provenance-v1', approval=approval,
                metadata=fusion.source_record(metadata_path), sidecar=sidecar_source,
                helper_source=fusion.source_record(h3_video_sft.__file__),
                validator_source=fusion.source_record(fusion.__file__), caption_version=caption_version,
                train_ids=[row['id'] for row in training_rows], val_ids=[row['id'] for row in val_rows],
                approval_chain_verified=True, fixture_only=False, formal_reuse_forbidden=False)


def validate_data_boundary(args):
    scope = getattr(args, 'data_scope', 'interface')
    if scope not in ('interface', 'reviewed_pilot', 'reviewed_full', 'source_label_pilot', 'source_label_full'):
        raise ValueError('Unknown explicit data scope')
    if scope == 'interface':
        if args.smoke_steps != 2:
            raise ValueError('Interface data requires --smoke-steps 2; formal training needs an explicit reviewed scope')
    elif args.smoke_steps != 0:
        raise ValueError('Reviewed data requires smoke_steps=0')
    elif scope in ('reviewed_pilot', 'source_label_pilot', 'source_label_full') and (args.min_epochs != 3 or args.max_epochs != 3):
        raise ValueError('Pilot requires exactly three full epochs')
    elif scope == 'reviewed_full' and (args.min_epochs != 3 or args.max_epochs != 10):
        raise ValueError('Reviewed full requires min_epochs=3/max_epochs=10')
    rows, training_rows, val_rows = read_metadata(args.metadata)
    if scope == 'interface':
        return rows, training_rows, val_rows, dict(protocol='omni-interface-smoke-only-v1',
            data_scope=scope, approval_chain_verified=False, formal_reuse_forbidden=True)
    required_counts = (512, 128) if scope in ('reviewed_pilot', 'source_label_pilot') else (6000, 500)
    if (len(training_rows), len(val_rows)) != required_counts:
        raise ValueError(f'{scope} requires exact train/val counts {required_counts}')
    versions = {row.get('caption_version') for row in rows if isinstance(row.get('caption_version'), str)}
    if (len(versions) != 1 or any(row.get('caption_version') not in versions or not row['id'] or
            any(row.get(key, False) is not False for key in ('fixture_only', 'formal_reuse_forbidden')) for row in rows)):
        raise ValueError('Formal metadata requires one caption_version, nonempty IDs and no fixture flags')
    if scope == 'source_label_full':
        from omni_source_label_full import source_label_full_provenance
        validator = source_label_full_provenance
    else:
        validator = source_label_omni_provenance if scope == 'source_label_pilot' else formal_omni_provenance
    data = validator(args.metadata, next(iter(versions)), rows, training_rows, val_rows)
    data['data_scope'] = scope
    return rows, training_rows, val_rows, data


def main_process_result(operation, rank, world):
    shared = [None]
    if rank == 0:
        try:
            shared[0] = dict(result=operation())
        except Exception as error:
            shared[0] = dict(error=f'{type(error).__name__}: {error}')
    if world > 1:
        dist.broadcast_object_list(shared, src=0)
    if 'error' in shared[0]:
        raise ValueError(shared[0]['error'])
    return shared[0]['result']


def provenance(args, rows, world, data_provenance):
    import peft
    import transformers
    model_files = {}
    # SHA weights too: path/config names alone do not identify a training base.
    for p in sorted(args.model.rglob('*')):
        if p.is_file() and p.suffix in ('.json', '.safetensors', '.bin', '.model', '.jinja'):
            model_files[str(p.relative_to(args.model))] = file_sha(p)
    if not any(k.endswith(('.safetensors', '.bin')) for k in model_files):
        raise ValueError('No local pretrained weight files')
    audio_files = {r['audios'][0]: file_sha(r['audios'][0]) for r in rows}
    recipe = {k: v for k, v in vars(args).items() if k not in ('out', 'resume', 'metadata', 'model', 'command')}
    payload = dict(schema=SCHEMA, source_sha256=file_sha(__file__), metadata_sha256=file_sha(args.metadata),
                   model_path=str(args.model.resolve()), model_id=args.model_id, model_files=model_files,
                   audio_files=audio_files, data_provenance=data_provenance, recipe=recipe, world_size=world,
                   torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__)
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload, fingerprint


@torch.no_grad()
def evaluate(model, processor, rows, args, device, dtype, rank, world, n_window):
    model.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    for row in rows[rank::world]:
        inputs = to_device(prepare_inputs(processor, row, max_length=args.max_length, audio_n_window=n_window), device, dtype)
        labels = inputs.pop('labels')
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == 'cuda'):
            output = model(**inputs, use_cache=False)
        nll, tokens = shifted_nll(output.logits, labels)
        totals += torch.stack((nll.detach().double(), tokens.double()))
    if world > 1:
        dist.all_reduce(totals)
    if not torch.isfinite(totals).all() or totals[1] <= 0:
        raise FloatingPointError('Invalid token-weighted validation NLL')
    samples = []
    if rank == 0:
        for row in rows[:args.val_generate]:
            inputs = to_device(prepare_inputs(processor, row, training=False, max_length=args.max_length,
                                             audio_n_window=n_window), device, dtype)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == 'cuda'):
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True,
                                       eos_token_id=processor.tokenizer.eos_token_id,
                                       pad_token_id=processor.tokenizer.pad_token_id)
            if not isinstance(generated, torch.Tensor):
                raise ValueError('Expected text-only Thinker generation')
            samples.append(dict(id=row['id'], caption=processor.batch_decode(
                generated[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)[0]))
    if world > 1:
        shared = [samples]
        dist.broadcast_object_list(shared, src=0)
        samples = shared[0]
    model.train()
    return dict(nll=float(totals[0] / totals[1]), assistant_tokens=int(totals[1]), samples=samples)


def early_stop(state, val_nll, min_epochs, patience, delta):
    if state['best_nll'] is None or val_nll < state['best_nll'] * (1 - delta):
        state['best_nll'] = val_nll
        state['stale_epochs'] = 0
    else:
        state['stale_epochs'] += 1
    return state['epoch'] >= min_epochs and state['stale_epochs'] >= patience


def train(args):
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if not torch.cuda.is_available():
        raise RuntimeError('Pretrained training requires CUDA; use selftest for an explicit CPU fixture')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 is required for this training recipe')
    if world > 1:
        dist.init_process_group('nccl')
    elif args.zero_stage1:
        raise ValueError('--zero-stage1 requires torchrun WORLD_SIZE > 1')
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    def preflight():
        if not args.out.parent.is_dir() or not args.model.is_dir():
            raise ValueError('Require existing output parent and local model directory')
        if args.resume:
            if not args.out.is_dir() or args.resume.resolve().parent != args.out.resolve():
                raise ValueError('Resume checkpoint must belong to the existing output directory')
            if (args.out / 'complete.json').exists():
                raise ValueError('Cannot resume an already completed run')
        elif args.out.exists():
            raise FileExistsError(f'Output already exists: {args.out}')
        rows, training_rows, val_rows, data = validate_data_boundary(args)
        payload, fingerprint = provenance(args, rows, world, data)
        if args.resume:
            if json.loads((args.out / 'provenance.json').read_text()) != payload:
                raise ValueError('Run provenance/approval identity changed; refusing incomplete resume')
            marker = json.loads((args.resume / 'complete.json').read_text())
            if marker.get('schema') != SCHEMA or marker.get('fingerprint') != fingerprint:
                raise ValueError('Resume checkpoint provenance/approval identity changed before model loading')
        return rows, training_rows, val_rows, payload, fingerprint
    try:
        rows, training_rows, val_rows, payload, fingerprint = main_process_result(preflight, rank, world)
        if not args.resume:
            def create_output():
                args.out.mkdir(exist_ok=False)
                atomic_json(args.out / 'provenance.json', payload)
            main_process_result(create_output, rank, world)
    except Exception:
        if world > 1:
            dist.destroy_process_group()
        raise
    family = identify_family(args.model)
    processor_cls, config_cls, model_cls = model_classes(family)
    processor = processor_cls.from_pretrained(args.model, local_files_only=True)
    if processor.tokenizer.pad_token_id is None or processor.tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer must define distinct assistant EOS and padding semantics')
    # Explicit Thinker config extraction handles old Qwen2_5OmniModel architectures.
    config = config_cls.from_pretrained(args.model, local_files_only=True)
    config.text_config.use_cache = False
    config.text_config.output_router_logits = False
    base, loading = model_cls.from_pretrained(args.model, config=config, dtype=torch.bfloat16,
                    device_map={'': device}, attn_implementation=args.attention,
                    local_files_only=True, output_loading_info=True)
    if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise ValueError(f'Incomplete Thinker pretrained load: {loading}')
    unexpected = loading.get('unexpected_keys', [])
    if any(not k.startswith(('talker.', 'code2wav.', 'token2wav.')) for k in unexpected):
        raise ValueError(f'Unexpected non-speech checkpoint keys: {unexpected}')
    log(dict(event='pretrained_loaded', family=family, discarded_speech_keys=len(unexpected)), rank)
    model, scope = attach_lora(base, family, args.lora_rank, args.lora_alpha)
    visual_hook = forbid_visual(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    n_window = config.audio_config.n_window if family == 'qwen3' else None
    log(dict(event='training_scope', **scope), rank)
    if rank == 0 and not args.resume:
        processor.save_pretrained(args.out / 'processor')
        atomic_json(args.out / 'scope.json', scope)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    optimizer = make_optimizer(model, args.lora_lr, args.projector_lr, args.zero_stage1)
    state = dict(epoch=0, position=0, permutation=None, updates=0, examples=0, tokens=0,
                 best_nll=None, stale_epochs=0, history=[], padded_forwards=0)
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, fingerprint, device, rank, world)
    validate_cursor(state, len(training_rows), args.max_epochs)
    wrapped = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                find_unused_parameters=True, broadcast_buffers=False,
                gradient_as_bucket_view=True) if world > 1 else model
    wrapped.train()
    chunk_size = world * args.accumulation
    stop_reason = None
    checkpoint = args.resume
    if checkpoint is None:
        checkpoint = save_checkpoint(args.out, model, optimizer, state, fingerprint, device, rank, world, args.zero_stage1)
    boundary_rng = rng_state(device)
    optimizer_boundary_safe = True
    start_time = time.monotonic()
    initial_hash = tensor_state_hash(get_peft_model_state_dict(model)) if args.smoke_steps else None
    try:
        while state['epoch'] < args.max_epochs:
            if state['permutation'] is None:
                generator = torch.Generator().manual_seed(args.seed + state['epoch'])
                state['permutation'] = torch.randperm(len(training_rows), generator=generator).tolist()
            permutation = state['permutation']
            while state['position'] < len(permutation):
                boundary_rng = rng_state(device)
                end = min(state['position'] + chunk_size, len(permutation))
                global_indices = permutation[state['position']:end]
                local_indices = global_indices[rank::world]
                # All ranks take identical numbers of forwards/backwards; zero-loss tail
                # forwards are explicitly excluded from examples/tokens, never duplicated supervision.
                forwards = math.ceil(len(global_indices) / world)
                batches = []
                local_tokens = 0
                for slot in range(forwards):
                    active = slot < len(local_indices)
                    idx = local_indices[slot] if active else global_indices[0]
                    inputs = prepare_inputs(processor, training_rows[idx], max_length=args.max_length, audio_n_window=n_window)
                    if active:
                        local_tokens += int((inputs['labels'][:, 1:] != -100).sum())
                    else:
                        inputs['labels'].fill_(-100)
                    batches.append((inputs, active))
                token_count = torch.tensor(local_tokens, dtype=torch.long, device=device)
                if world > 1:
                    dist.all_reduce(token_count)
                if token_count <= 0:
                    raise ValueError('Optimizer window contains no assistant target tokens')
                optimizer.zero_grad(set_to_none=True)
                local_nll = torch.zeros((), dtype=torch.float64, device=device)
                for slot, (inputs, active) in enumerate(batches):
                    sync = wrapped.no_sync() if world > 1 and slot + 1 < forwards else nullcontext()
                    with sync:
                        inputs = to_device(inputs, device, torch.bfloat16)
                        labels = inputs.pop('labels')
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            output = wrapped(**inputs, use_cache=False)
                        nll, _ = shifted_nll(output.logits, labels)
                        if not torch.isfinite(nll):
                            raise FloatingPointError('Nonfinite training NLL')
                        (nll * (world / int(token_count))).backward()
                        local_nll += nll.detach().double()
                        del output, nll, labels, inputs
                report = gradient_report(model)
                if state['updates'] == 0 or args.smoke_steps:
                    log(dict(event='gradient_scope', update=state['updates'] + 1, **report), rank)
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                     args.grad_clip, error_if_nonfinite=True)
                optimizer_boundary_safe = False
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                state['position'] = end
                state['updates'] += 1
                state['examples'] += len(global_indices)
                state['tokens'] += int(token_count)
                state['padded_forwards'] += forwards * world - len(global_indices)
                optimizer_boundary_safe = True
                boundary_rng = rng_state(device)
                if world > 1:
                    dist.all_reduce(local_nll)
                log(dict(event='update', updates=state['updates'], epoch=state['epoch'], position=end,
                         examples=state['examples'], assistant_tokens=state['tokens'],
                         effective_batch=len(global_indices), configured_batch=chunk_size,
                         nll=float(local_nll / token_count), grad_norm=float(norm),
                         assistant_tokens_per_second=state['tokens'] / (time.monotonic() - start_time)), rank)
                if args.smoke_steps and state['updates'] >= args.smoke_steps:
                    stop_reason = 'smoke_updates_only_not_completed_sft'
                    break
                if args.save_every and state['updates'] % args.save_every == 0 and end < len(permutation):
                    checkpoint = save_checkpoint(args.out, model, optimizer, state, fingerprint, device, rank, world, args.zero_stage1)
            if stop_reason:
                break
            validation = evaluate(model, processor, val_rows, args, device, torch.bfloat16, rank, world, n_window)
            state['epoch'] += 1
            state['position'] = 0
            state['permutation'] = None
            boundary_rng = rng_state(device)
            should_stop = early_stop(state, validation['nll'], args.min_epochs, args.patience, args.relative_delta)
            state['history'].append(dict(epoch=state['epoch'], updates=state['updates'], examples=state['examples'],
                                         **validation))
            log(dict(event='full_epoch_validation', **state['history'][-1], stale_epochs=state['stale_epochs']), rank)
            checkpoint = save_checkpoint(args.out, model, optimizer, state, fingerprint, device, rank, world, args.zero_stage1)
            if should_stop:
                stop_reason = 'validation_plateau_not_human_quality_acceptance'
                break
        if stop_reason is None:
            stop_reason = 'epoch_cap_convergence_unconfirmed'
        if checkpoint is None or json.loads((checkpoint / 'complete.json').read_text())['updates'] != state['updates']:
            checkpoint = save_checkpoint(args.out, model, optimizer, state, fingerprint, device, rank, world, args.zero_stage1)
        trained_hash = tensor_state_hash(get_peft_model_state_dict(model))
        if args.smoke_steps:
            if state['updates'] != args.smoke_steps or trained_hash == initial_hash:
                raise ValueError('Smoke did not perform exactly the requested real updates')
            restored = load_checkpoint(checkpoint, model, optimizer, fingerprint, device, rank, world)
            if restored != state or tensor_state_hash(get_peft_model_state_dict(model)) != trained_hash:
                raise ValueError('Full smoke checkpoint reload failed')
            deployed = processor_cls.from_pretrained(args.out / 'processor', local_files_only=True)
            left = prepare_inputs(processor, training_rows[0], training=False, audio_n_window=n_window)
            right = prepare_inputs(deployed, training_rows[0], training=False, audio_n_window=n_window)
            if set(left) != set(right) or any(not torch.equal(left[k], right[k]) for k in left):
                raise ValueError('Saved inference processor differs from training processor')
        posttrain_predictions = []
        if args.infer:
            # Inference always uses the saved deployment processor and never passes labels.
            deployment = processor_cls.from_pretrained(args.out / 'processor', local_files_only=True)
            model.eval()
            if rank == 0:
                with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    for row in val_rows[:args.val_generate]:
                        inputs = to_device(prepare_inputs(deployment, row, training=False,
                            max_length=args.max_length, audio_n_window=n_window), device, torch.bfloat16)
                        if 'labels' in inputs:
                            raise ValueError('Deployment input unexpectedly contains training labels')
                        generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                                   eos_token_id=deployment.tokenizer.eos_token_id,
                                                   pad_token_id=deployment.tokenizer.pad_token_id)
                        if not isinstance(generated, torch.Tensor):
                            raise ValueError('Thinker must produce text only')
                        posttrain_predictions.append(dict(id=row['id'], caption=deployment.batch_decode(
                            generated[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)[0]))
                atomic_json(args.out / 'posttrain_predictions.json', dict(checkpoint=checkpoint.name,
                    predictions=posttrain_predictions, human_review=False))
            if world > 1:
                dist.barrier()
        result = dict(schema=SCHEMA, fingerprint=fingerprint, stop_reason=stop_reason,
                      completed_sft=not bool(args.smoke_steps), smoke_verified=bool(args.smoke_steps),
                      data_scope=payload['data_provenance']['data_scope'],
                      label_quality=payload['data_provenance'].get('label_quality',
                          'reviewed' if payload['data_provenance']['approval_chain_verified'] else 'interface_only'),
                      approval_chain_verified=payload['data_provenance']['approval_chain_verified'],
                      posttrain_predictions=posttrain_predictions,
                      checkpoint=checkpoint.name, updates=state['updates'], full_epochs=state['epoch'],
                      examples=state['examples'], assistant_tokens=state['tokens'], history=state['history'],
                      padded_forwards=state['padded_forwards'], trained_adapter_sha256=trained_hash,
                      human_quality_accepted=False)
        if rank == 0:
            atomic_json(args.out / 'complete.json', result)
        log(dict(event='run_complete', **result), rank)
    except (Exception, KeyboardInterrupt) as error:
        optimizer.zero_grad(set_to_none=True)
        saved_error = None
        # A single-rank failure cannot safely enter distributed collectives. Distributed
        # recovery uses the last fully committed checkpoint, never incomplete rank shards.
        if world == 1 and optimizer_boundary_safe:
            restore_rng(boundary_rng, device)
            name = f'checkpoint-u{state["updates"]:07d}-e{state["epoch"]:02d}-p{state["position"]:07d}'
            if not (args.out / name / 'complete.json').is_file():
                try:
                    checkpoint = save_checkpoint(args.out, model, optimizer, state, fingerprint, device, rank, world)
                except Exception as checkpoint_error:
                    saved_error = repr(checkpoint_error)
        if rank == 0:
            atomic_json(args.out / 'partial.json', dict(error=repr(error), updates=state['updates'],
                        epoch=state['epoch'], position=state['position'], optimizer_boundary_safe=optimizer_boundary_safe,
                        checkpoint_save_error=saved_error, resume_from='latest.json if present',
                        distributed_recovery='last atomic complete checkpoint only' if world > 1 else None))
        raise
    finally:
        visual_hook.remove()
        if world > 1:
            dist.destroy_process_group()


def inference(args):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Inference is single-process; shard explicitly outside this entry point')
    rows, _, _ = read_metadata(args.metadata, training=False)
    family = identify_family(args.model)
    processor_cls, config_cls, model_cls = model_classes(family)
    processor_path = args.checkpoint.parent / 'processor' if args.checkpoint else args.model
    processor = processor_cls.from_pretrained(processor_path, local_files_only=True)
    config = config_cls.from_pretrained(args.model, local_files_only=True)
    model = model_cls.from_pretrained(args.model, config=config, dtype=torch.bfloat16,
                     device_map='cuda', local_files_only=True, attn_implementation=args.attention)
    if args.checkpoint:
        complete = json.loads((args.checkpoint / 'complete.json').read_text())
        for name, expected in complete['files'].items():
            if file_sha(args.checkpoint / name) != expected:
                raise ValueError(f'Corrupt checkpoint: {name}')
        identity = json.loads((args.checkpoint.parent / 'provenance.json').read_text())
        if str(args.model.resolve()) != identity['model_path'] or args.model_id != identity['model_id']:
            raise ValueError('Inference base path/model ID mismatch')
        for name, expected in identity['model_files'].items():
            if file_sha(args.model / name) != expected:
                raise ValueError(f'Inference base weight/protocol mismatch: {name}')
        model = PeftModel.from_pretrained(model, args.checkpoint / 'adapter', is_trainable=False)
    else:
        # Still forbid visual calls with the same boundary as training.
        model = model.eval().requires_grad_(False)
    base = model.get_base_model() if isinstance(model, PeftModel) else model
    if hasattr(base, 'talker') or hasattr(base, 'code2wav'):
        raise ValueError('Speech modules must not be created')
    base.visual.register_forward_pre_hook(lambda *a, **k: (_ for _ in ()).throw(RuntimeError('Visual tower called')))
    model.eval().requires_grad_(False)
    selected = [r for r in rows if r['id'] in args.ids] if args.ids else [r for r in rows if r['split'] == 'val'][:3]
    if not selected or len(selected) > 3 or args.ids and set(args.ids) != {r['id'] for r in selected}:
        raise ValueError('Select one to three existing IDs; no implicit full evaluation')
    if not args.out.parent.is_dir() or args.out.exists():
        raise ValueError('Inference output must be a new file in an existing directory')
    predictions = []
    n_window = config.audio_config.n_window if family == 'qwen3' else None
    with torch.inference_mode():
        for row in selected:
            inputs = to_device(prepare_inputs(processor, row, training=False, max_length=args.max_length,
                                audio_n_window=n_window), model.device, torch.bfloat16)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                           eos_token_id=processor.tokenizer.eos_token_id,
                                           pad_token_id=processor.tokenizer.pad_token_id)
            if not isinstance(generated, torch.Tensor):
                raise ValueError('Thinker must output text only')
            predictions.append(dict(id=row['id'], audio_path=row['audios'][0],
                audio_sha256=file_sha(row['audios'][0]), audio_window_seconds=row.get('audio_window_seconds'),
                caption=processor.batch_decode(generated[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)[0]))
    atomic_json(args.out, dict(model_id=args.model_id, checkpoint=str(args.checkpoint) if args.checkpoint else None,
                             predictions=predictions, human_review=False))


def processor_test(args):
    """CPU-only real local processor and real waveform, no pretrained model loading."""
    if not args.out.parent.is_dir() or args.out.exists():
        raise ValueError('Processor test requires new JSON result in an existing parent')
    torch.set_num_threads(2)
    family = identify_family(args.model)
    processor_cls, config_cls, _ = model_classes(family)
    processor = processor_cls.from_pretrained(args.model, local_files_only=True)
    config = config_cls.from_pretrained(args.model, local_files_only=True)
    row = dict(id='real-processor-boundary', audios=[str(args.audio.resolve())],
               audio_window_seconds=args.window,
               messages=[dict(role='user', content='<audio>Write a continuous dance choreography plan suitable for this music. '
                                'Describe motion and transitions only; do not infer appearance or scene.'),
                         dict(role='assistant', content='The dancer sways smoothly, then steps sideways and turns, '
                                'linking broad arm sweeps with a grounded rhythmic weight shift.')])
    waveform = read_audio(row)
    n_window = config.audio_config.n_window if family == 'qwen3' else None
    training = prepare_inputs(processor, row, waveform=waveform, audio_n_window=n_window, max_length=8192)
    deploying = prepare_inputs(processor, row, training=False, waveform=waveform, audio_n_window=n_window, max_length=8192)
    count = int((training['labels'] != -100).sum())
    audio_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_token)
    eos_id = processor.tokenizer.eos_token_id
    supervised = training['labels'][training['labels'] != -100]
    if 'labels' in deploying or not torch.equal(training['input_ids'][0, :deploying['input_ids'].shape[-1]],
                                                deploying['input_ids'][0]):
        raise ValueError('Inference/training processed prefix mismatch')
    if int((supervised == eos_id).sum()) != 1 or int((training['labels'][training['input_ids'] == audio_id] != -100).sum()):
        raise ValueError('EOS/audio supervision boundary failed')
    with tempfile.TemporaryDirectory(prefix='real-omni-processor-cpu-') as directory:
        processor.save_pretrained(directory)
        reloaded = processor_cls.from_pretrained(directory, local_files_only=True)
        restored = prepare_inputs(reloaded, row, training=False, waveform=waveform, audio_n_window=n_window, max_length=8192)
        if set(restored) != set(deploying) or any(not torch.equal(restored[k], deploying[k]) for k in restored):
            raise ValueError('Saved deployment processor parity failed')
    result = dict(complete=True, cpu_only=True, pretrained_model_loaded=False, fixture_caption=True,
                  family=family, model_path=str(args.model.resolve()), model_config_sha256=file_sha(args.model / 'config.json'),
                  source_sha256=file_sha(__file__), audio_path=str(args.audio.resolve()), audio_sha256=file_sha(args.audio),
                  audio_window_seconds=args.window, waveform_sr=16000, waveform_samples=len(waveform),
                  waveform_sha256=hashlib.sha256(waveform.tobytes()).hexdigest(),
                  input_features_shape=list(training['input_features'].shape),
                  feature_attention_mask_shape=list(training['feature_attention_mask'].shape),
                  valid_audio_feature_frames=int(training['feature_attention_mask'].sum()),
                  total_tokens=int(training['input_ids'].shape[-1]),
                  processed_prefix_tokens=int(deploying['input_ids'].shape[-1]),
                  audio_placeholder_tokens=int((training['input_ids'] == audio_id).sum()),
                  supervised_caption_eos_tokens=count, shifted_loss_tokens=int((training['labels'][:, 1:] != -100).sum()),
                  user_audio_padding_supervised_tokens=0, supervised_eos_tokens=1,
                  saved_processor_inference_parity=True, inference_contains_labels=False)
    atomic_json(args.out, result)
    log(result)


def selftest():
    """Tiny random *real architecture* models and explicit synthetic processor fixture."""
    torch.set_num_threads(2)
    torch.manual_seed(17)
    from transformers import (Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration,
                              Qwen3OmniMoeThinkerConfig, Qwen3OmniMoeThinkerForConditionalGeneration)

    class TokenizerFixture:
        eos_token_id = 2
        pad_token_id = 0
        def convert_tokens_to_ids(self, token):
            return 3

    class ProcessorFixture:
        audio_token = '<|audio|>'
        tokenizer = TokenizerFixture()
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return 'prefix' if add_generation_prompt else 'full'
        def __call__(self, text, **kwargs):
            assert kwargs['audio_kwargs']['sampling_rate'] == 16000
            assert kwargs['audio'][0].shape == (1600,)
            ids = [4, 3, 3, 5] if text == 'prefix' else [4, 3, 3, 5, 6, 7, 2, 8]
            return dict(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                        input_features=torch.zeros(1, 8, 100), feature_attention_mask=torch.ones(1, 100, dtype=torch.long))

    row = dict(id='explicit-fixture', audios=['fixture-never-read.wav'],
               messages=[dict(role='user', content='<audio>Plan continuous dancing.'),
                         dict(role='assistant', content='Sway and turn smoothly.')])
    fixture = prepare_inputs(ProcessorFixture(), row, waveform=np.zeros(1600, dtype='float32'))
    assert fixture['labels'].tolist() == [[-100, -100, -100, -100, 6, 7, 2, -100]]
    inference_row = copy.deepcopy(row)
    inference_row['messages'] = inference_row['messages'][:-1]
    inferred = prepare_inputs(ProcessorFixture(), inference_row, training=False,
                              waveform=np.zeros(1600, dtype='float32'))
    assert inferred['input_ids'].tolist() == [[4, 3, 3, 5]] and 'labels' not in inferred
    padded = {k: v.clone() for k, v in fixture.items()}
    padded['input_ids'] = F.pad(padded['input_ids'], (2, 2), value=0)
    padded['attention_mask'] = F.pad(padded['attention_mask'], (2, 2), value=0)
    prefix = dict(input_ids=torch.tensor([[4, 3, 3, 5]]), attention_mask=torch.ones(1, 4, dtype=torch.long))
    assert mask_response(padded, prefix, 2).tolist() == [[-100, -100, -100, -100, -100, -100, 6, 7, 2, -100, -100, -100]]
    bad = copy.deepcopy(row)
    bad['messages'][0]['content'] += ' Sway and turn smoothly.'
    try:
        conversation(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Caption leakage was accepted')
    logits = torch.randn(1, 8, 12)
    nll, count = shifted_nll(logits, fixture['labels'])
    expected = sum(F.cross_entropy(logits[:, i - 1], torch.tensor([t])) for i, t in ((4, 6), (5, 7), (6, 2)))
    assert count == 3 and torch.allclose(nll, expected)
    reports = []
    for family, config_cls, model_cls in [('qwen2.5', Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration),
                                         ('qwen3', Qwen3OmniMoeThinkerConfig, Qwen3OmniMoeThinkerForConditionalGeneration)]:
        audio = dict(num_mel_bins=8, encoder_layers=5, encoder_attention_heads=2, encoder_ffn_dim=32,
                     d_model=16, output_dim=32, max_source_positions=100, n_window=50)
        vision = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, patch_size=2,
                      spatial_merge_size=1, temporal_patch_size=1, out_hidden_size=32)
        text = dict(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=1,
                    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                    pad_token_id=0, eos_token_id=2, rope_parameters=dict(rope_type='default',
                    rope_theta=10000., mrope_section=[1, 1, 2]), tie_word_embeddings=False)
        if family == 'qwen3':
            audio.update(downsample_hidden_size=8, n_window_infer=100, conv_chunksize=100)
            vision.update(num_position_embeddings=16, deepstack_visual_indexes=[])
            text.update(num_experts=2, num_experts_per_tok=2, moe_intermediate_size=16,
                        rope_parameters=dict(rope_type='default', rope_theta=10000., mrope_section=[1, 1, 2],
                                             mrope_interleaved=True))
            config = config_cls(audio_config=audio, vision_config=vision, text_config=text,
                                audio_token_id=3, audio_start_token_id=9, image_token_id=10, video_token_id=11)
        else:
            vision.update(fullatt_block_indexes=[0], window_size=4)
            config = config_cls(audio_config=audio, vision_config=vision, text_config=text,
                                audio_token_index=3, audio_start_token_id=9, audio_end_token_id=12,
                                image_token_index=10, video_token_index=11)
        config.vision_start_token_id = 10
        config._attn_implementation = 'eager'
        base = model_cls(config)
        base.eval()  # from_pretrained returns an eval-mode base inside the new PEFT wrapper.
        pristine_base = copy.deepcopy(base.state_dict())
        model, scope = attach_lora(base, family, rank=2, alpha=4)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        decoder = model.get_base_model().model.layers[0]
        assert decoder.gradient_checkpointing and not decoder.training
        checkpoint_calls = []
        checkpoint_function = decoder._gradient_checkpointing_func
        def observed_checkpoint(function, *args, **kwargs):
            checkpoint_calls.append(True)
            return checkpoint_function(function, *args, **kwargs)
        decoder._gradient_checkpointing_func = observed_checkpoint
        assert scope['audio_last_four'] == [1, 2, 3, 4]
        assert not any('audio_tower.layers.0.' in p['name'] for p in scope['trainable'])
        frozen_initial = tensor_state_hash({n: p for n, p in model.named_parameters() if not p.requires_grad})
        hook = forbid_visual(model)
        optimizer = make_optimizer(model, 5e-5, 1e-5)
        # 100 mel frames map to 25 (2.5) or 13 (3) actual audio embeddings.
        audio_tokens = 25 if family == 'qwen2.5' else 13
        ids = torch.tensor([[4, 9] + [3] * audio_tokens + [5, 6, 7, 2]])
        labels = torch.full_like(ids, -100)
        labels[0, -3:] = ids[0, -3:]
        inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                      input_features=torch.randn(1, 8, 100), feature_attention_mask=torch.ones(1, 100, dtype=torch.long))
        initial = tensor_state_hash(get_peft_model_state_dict(model))
        for update in range(2):
            model.train()
            output = model(**inputs, use_cache=False)
            loss, tokens = shifted_nll(output.logits, labels)
            (loss / tokens).backward()
            gradients = gradient_report(model)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        assert decoder.training and len(checkpoint_calls) == 2
        trained = tensor_state_hash(get_peft_model_state_dict(model))
        assert initial != trained
        assert frozen_initial == tensor_state_hash({n: p for n, p in model.named_parameters() if not p.requires_grad})
        state = dict(epoch=0, position=2, permutation=[1, 0], updates=2, examples=2, tokens=6,
                     history=[], best_nll=None, stale_epochs=0, padded_forwards=0)
        validate_cursor(state, 2, 10)
        bad_cursor = copy.deepcopy(state)
        bad_cursor['permutation'] = [0, 0]
        try:
            validate_cursor(bad_cursor, 2, 10)
        except ValueError:
            pass
        else:
            raise AssertionError('Corrupt sampler resume accepted')
        with tempfile.TemporaryDirectory(prefix='omni-sft-cpu-fixture-') as directory:
            checkpoint = save_checkpoint(Path(directory), model, optimizer, state, 'explicit-fixture',
                                         torch.device('cpu'), 0, 1)
            from safetensors.torch import load_file
            saved_adapter = load_file(checkpoint / 'adapter' / 'adapter_model.safetensors')
            assert all('lora_' in n or any(projector in n for projector in scope['projectors']) for n in saved_adapter)
            assert not any('embed_tokens' in n or 'lm_head' in n for n in saved_adapter)
            expected_random = (random.random(), np.random.rand(), torch.rand(1))
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:
                        p.add_(1.)
            restored = load_checkpoint(checkpoint, model, optimizer, 'explicit-fixture', torch.device('cpu'), 0, 1)
            actual_random = (random.random(), np.random.rand(), torch.rand(1))
            assert restored == state and tensor_state_hash(get_peft_model_state_dict(model)) == trained
            assert actual_random[:2] == expected_random[:2] and torch.equal(actual_random[2], expected_random[2])
            model.eval()
            with torch.no_grad():
                reference = model(**inputs, use_cache=False).logits
            clone_base = model_cls(config)
            clone_base.load_state_dict(pristine_base, strict=True)
            cloned = PeftModel.from_pretrained(clone_base, checkpoint / 'adapter', is_trainable=True)
            clone_optimizer = make_optimizer(cloned, 5e-5, 1e-5)
            load_checkpoint(checkpoint, cloned, clone_optimizer, 'explicit-fixture', torch.device('cpu'), 0, 1)
            cloned.eval()
            with torch.no_grad():
                assert torch.equal(reference, cloned(**inputs, use_cache=False).logits)
            for candidate, opt in ((model, optimizer), (cloned, clone_optimizer)):
                candidate.train()
                out = candidate(**inputs, use_cache=False)
                loss, tokens = shifted_nll(out.logits, labels)
                (loss / tokens).backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            assert tensor_state_hash(get_peft_model_state_dict(model)) == tensor_state_hash(get_peft_model_state_dict(cloned))
        hook.remove()
        reports.append(dict(family=family, updates=2, nonzero_by_scope=gradients['nonzero_by_scope'],
                            fused_mlp_parameters=len(scope['target_parameters']),
                            pretrained_eval_mode_reproduced=True,
                            actual_gradient_checkpoint_calls=len(checkpoint_calls),
                            checkpoint_rng_optimizer_continuation_exact=True))
    state = dict(epoch=3, best_nll=1., stale_epochs=2)
    assert early_stop(state, .997, 3, 3, .005)
    log(dict(event='selftest_passed', fixture_only=True, pretrained_smoke=False,
             labels_padding_eos_shift=True, real_random_architecture_gradient_tests=reports))


def data_selftest(args):
    """CPU contract simulations only: no actual review, model, training or durable export."""
    import contextlib
    import io
    from unittest import mock
    import dance_caption_fusion as fusion
    import h3_video_sft
    torch.set_num_threads(2)
    if args.out and (not args.out.parent.is_dir() or args.out.exists()):
        raise ValueError('Data selftest output requires a new JSON file in an existing parent')
    checks = []
    def rejects(name, operation):
        try:
            operation()
        except (ValueError, KeyError, TypeError, FileNotFoundError):
            checks.append(dict(case=name, rejected=True))
        else:
            raise AssertionError(f'Accepted invalid data: {name}')
    def write_rows(path, rows):
        Path(path).write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with tempfile.TemporaryDirectory(prefix='omni-formal-gate-synthetic-CPU-') as directory:
        root = Path(directory)
        fixture = fusion.CpuChecks()
        exported = fixture.synthetic_export_inputs(root)
        review = fixture.assistant_review_simulation(fusion.read_json(exported.review), exported.manifest)
        atomic_json(exported.review, review)
        with contextlib.redirect_stdout(io.StringIO()):
            fusion.export_sft(exported)
        metadata = exported.out / 'omni.jsonl'
        rows, train_rows, val_rows = read_metadata(metadata)
        version = rows[0]['caption_version']
        with mock.patch.object(h3_video_sft, 'formal_training_provenance',
                               wraps=h3_video_sft.formal_training_provenance) as shared:
            data = formal_omni_provenance(metadata, version, rows, train_rows, val_rows)
            assert shared.call_count == 1 and data['approval_chain_verified'] is True
        checks.append(dict(case='real_shared_chain_1_1', passed=True,
                           synthetic_cpu_contract_fixture=True, actual_approval=False))
        # Real-chain corruption cases are restored after each check; no approval is forged.
        original = {path: path.read_bytes() for path in exported.out.iterdir() if path.is_file()}
        sidecar_path = exported.out / 'source_sidecar.json'
        for case in ('output_sha', 'fake_approval', 'evidence_sha', 'sidecar_fixture', 'partition_drift',
                     'wrong_metadata_path', 'wrong_version'):
            sidecar = fusion.read_json(sidecar_path)
            changed_review = None
            if case == 'output_sha':
                metadata.write_text(metadata.read_text() + '\n')
            elif case in ('fake_approval', 'evidence_sha'):
                changed_review = exported.review.read_bytes()
                tampered = fusion.read_json(exported.review)
                if case == 'fake_approval':
                    tampered['records'][0]['facts_approved'] = 'true'
                else:
                    tampered['records'][0]['review_evidence'][1]['sha256'] = '0' * 64
                atomic_json(exported.review, tampered)
                review_source = fusion.source_record(exported.review)
                sidecar['review'] = review_source
                for record, approval in zip(sidecar['records'], tampered['records']):
                    record['human_review'] = approval
                    record['review_source'] = review_source
                    record['approval_provenance']['review'] = review_source
                atomic_json(sidecar_path, sidecar)
            elif case == 'sidecar_fixture':
                sidecar['fixture_only'] = True
                atomic_json(sidecar_path, sidecar)
            elif case == 'partition_drift':
                swapped = copy.deepcopy(train_rows)
                swapped[0]['messages'][1]['content'] += ' altered'
                write_rows(exported.out / 'train.jsonl', swapped)
                sidecar['outputs']['train.jsonl'] = fusion.source_record(exported.out / 'train.jsonl')
                atomic_json(sidecar_path, sidecar)
            check_path = exported.out / 'train.jsonl' if case == 'wrong_metadata_path' else metadata
            check_version = version + '-wrong' if case == 'wrong_version' else version
            rejects('real_chain_' + case, lambda: formal_omni_provenance(
                check_path, check_version, rows, train_rows, val_rows))
            for path, value in original.items():
                path.write_bytes(value)
            if changed_review is not None:
                exported.review.write_bytes(changed_review)
        pilot_root = root / 'source-label-synthetic-1-1'
        pilot_root.mkdir()
        pilot_manifest_path = pilot_root / 'manifest.json'
        pilot_caps_path = pilot_root / 'caps.json'
        pilot_manifest = fusion.read_json(exported.manifest)
        caps = {}
        for record in pilot_manifest['records']:
            sid = record['id']
            ms_path = pilot_root / (sid + '.ms.json')
            source_ms = dict(sid=sid, ms=['A few seconds earlier, she raises her arms.',
                                       'In the second right after, he bends his knees.'])
            atomic_json(ms_path, source_ms)
            record['source_MS'] = source_ms
            record['provenance']['ms'] = fusion.source_record(ms_path)
            record['source_selected_caption'] = dict(dims=dict(genre='hip-hop'))
            caps[sid] = record['source_selected_caption']
        atomic_json(pilot_caps_path, caps)
        pilot_manifest.update(source_code=fusion.source_record(fusion.__file__),
                              source_inputs=dict(caps=fusion.source_record(pilot_caps_path)),
                              counts=dict(train=1, val=1))
        atomic_json(pilot_manifest_path, pilot_manifest)
        pilot_manifest_source = fusion.source_record(pilot_manifest_path)
        pilot_version = SOURCE_LABEL_PROTOCOL + ':' + pilot_manifest_source['sha256']
        pilot_rows = [dict(id=record['id'], source_group=record['source_group'], split=record['split'],
            caption_version=pilot_version,
            messages=[dict(role='user', content=fusion.SFT_INSTRUCTION + '\n<audio>'),
                      dict(role='assistant', content=source_label_caption(record))],
            audios=[record['audio_path']], audio_window_seconds=[0., 5.0625])
            for record in pilot_manifest['records']]
        pilot_metadata = pilot_root / 'omni.jsonl'
        write_rows(pilot_metadata, pilot_rows)
        pilot_sidecar_path = pilot_root / 'source_label_sidecar.json'
        pilot_sidecar = dict(protocol=SOURCE_LABEL_PROTOCOL, complete=True,
            label_quality='source_labels_unreviewed', authorization=SOURCE_LABEL_AUTHORIZATION,
            human_review_completed=False, semantic_approval=False,
            original_source_window_mapping='unknown', counts=dict(train=1, val=1),
            manifest=pilot_manifest_source, source_code=fusion.source_record(__file__),
            outputs={'omni.jsonl': fusion.source_record(pilot_metadata)})
        atomic_json(pilot_sidecar_path, pilot_sidecar)
        pilot_loaded = read_metadata(pilot_metadata)
        def check_pilot():
            return source_label_omni_provenance(pilot_metadata, pilot_version, *pilot_loaded)
        with mock.patch.object(h3_video_sft, 'formal_training_provenance',
                               side_effect=AssertionError('Weak labels must not enter reviewed H3 chain')) as shared:
            pilot_data = check_pilot()
            shared.assert_not_called()
        assert pilot_data['approval_chain_verified'] is False and pilot_data['semantic_approval'] is False
        assert pilot_rows[0]['messages'][1]['content'] == (
            "Dance style: hip-hop. The dancer raises the dancer's arms. the dancer bends the dancer's knees.")
        checks.append(dict(case='source_labels_real_sources_1_1', passed=True,
                           synthetic_cpu_contract_fixture=True, actual_approval=False,
                           approval_chain_verified=False, original_source_window_mapping='unknown'))
        for key, value in (('authorization', 'unknown'), ('label_quality', 'reviewed'),
                           ('human_review_completed', True), ('semantic_approval', True), ('approval_chain_verified', True),
                           ('original_source_window_mapping', 'verified'), ('fixture_only', True),
                           ('formal_reuse_forbidden', True), ('complete', 'true')):
            atomic_json(pilot_sidecar_path, dict(pilot_sidecar, **{key: value}))
            rejects('source_labels_wrong_' + key, check_pilot)
        for key in ('source_code', 'manifest'):
            corrupted = copy.deepcopy(pilot_sidecar)
            corrupted[key]['sha256'] = '0' * 64
            atomic_json(pilot_sidecar_path, corrupted)
            rejects('source_labels_wrong_sha_' + key, check_pilot)
        atomic_json(pilot_sidecar_path, pilot_sidecar)
        for text in ('She follows the music.', 'He wears a shirt.', 'The dancer bends the fingers.',
                     'At 3 seconds she turns.'):
            bad_record = dict(pilot_manifest['records'][0], source_MS=dict(ms=[text]))
            rejects('source_labels_unsafe_' + text, lambda: source_label_caption(bad_record))
        rejects('source_labels_unsafe_genre', lambda: source_label_caption(dict(
            pilot_manifest['records'][0], source_selected_caption=dict(dims=dict(genre='<|invented|>')))))
        pilot_args = argparse.Namespace(metadata=pilot_metadata, data_scope='source_label_pilot',
                                       smoke_steps=0, min_epochs=3, max_epochs=3)
        rejects('source_labels_no_1_1_count_upgrade', lambda: validate_data_boundary(pilot_args))
        for case, overrides in (('smoke', dict(smoke_steps=2)), ('max_epochs', dict(max_epochs=4)),
                                ('min_epochs', dict(min_epochs=4)), ('unknown_scope', dict(data_scope='unknown'))):
            rejects('source_labels_boundary_' + case, lambda overrides=overrides: validate_data_boundary(
                argparse.Namespace(**(vars(pilot_args) | overrides))))
        fake_budget = [dict(row, id=row['id'] + str(index)) for row, count in zip(pilot_rows, (24, 8))
                       for index in range(count)]
        write_rows(pilot_metadata, fake_budget)
        rejects('source_labels_no_24_8_count_upgrade', lambda: validate_data_boundary(pilot_args))
        write_rows(pilot_metadata, pilot_rows)
        small = argparse.Namespace(metadata=metadata, data_scope='reviewed_pilot', smoke_steps=0,
                                   min_epochs=3, max_epochs=3)
        rejects('formal_1_1_budget', lambda: validate_data_boundary(small))
        for scope, counts, maximum in (('reviewed_pilot', (512, 128), 3),
                                        ('reviewed_full', (6000, 500), 10)):
            trial = root / scope
            trial.mkdir()
            expected, records = [], []
            for base, count in zip(rows, counts):
                for index in range(count):
                    row = copy.deepcopy(base)
                    row['id'] = f'{scope}-{base["split"]}-{index:05d}'
                    expected.append(row)
                    records.append(dict(id=row['id'], split=row['split'], source_group=row['source_group'],
                        caption_version=version, motion_caption=row['messages'][1]['content'],
                        audio_source=dict(path=row['audios'][0]), audio_window_seconds=row['audio_window_seconds']))
            trial_metadata = trial / 'omni.jsonl'
            trial_sidecar = dict(records=records, caption_version=version, counts=dict(zip(('train', 'val'), counts)),
                                 synthetic_cpu_contract_fixture=True, outputs={})
            for name in ('omni.jsonl', 'train.jsonl', 'val.jsonl'):
                selected = expected if name == 'omni.jsonl' else [row for row in expected if row['split'] == name[:-6]]
                write_rows(trial / name, selected)
                trial_sidecar['outputs'][name] = fusion.source_record(trial / name)
            trial_sidecar['outputs']['h3_video_train.json'] = dict(path=str(trial / 'h3_video_train.json'))
            atomic_json(trial / 'source_sidecar.json', trial_sidecar)
            trial_args = argparse.Namespace(metadata=trial_metadata, data_scope=scope, smoke_steps=0,
                                            min_epochs=3, max_epochs=maximum)
            with mock.patch.object(h3_video_sft, 'formal_training_provenance', return_value=dict(
                    approval_chain_verified=True, synthetic_mock=True, actual_approval=False)) as shared:
                with mock.patch.dict(globals(), formal_omni_provenance=mock.Mock(wraps=formal_omni_provenance)):
                    strict = globals()['formal_omni_provenance']
                    result = validate_data_boundary(trial_args)
                    assert strict.call_count == 1 and shared.call_count == 1
                assert (len(result[1]), len(result[2])) == counts
                shared.assert_called_with(str(trial / 'h3_video_train.json'), version)
                checks.append(dict(case=scope + '_exact_boundary', passed=True, counts=list(counts),
                                   strict_omni_helper_called=True, shared_h3_validator_mocked=True,
                                   synthetic_mock=True, actual_approval=False))
                if scope == 'reviewed_pilot':
                    for case in ('caption', 'version', 'source', 'window', 'split', 'fixture',
                                 'fixture_nonboolean', 'id', 'instruction'):
                        corrupted = copy.deepcopy(expected)
                        if case == 'caption':
                            corrupted[0]['messages'][1]['content'] += ' changed'
                        elif case == 'version':
                            corrupted[0]['caption_version'] += '-wrong'
                        elif case == 'source':
                            corrupted[0]['audios'] = [str(exported.manifest)]
                        elif case == 'window':
                            corrupted[0]['audio_window_seconds'] = [0, 4]
                        elif case == 'split':
                            corrupted[0]['split'] = 'val'
                        elif case in ('fixture', 'fixture_nonboolean'):
                            corrupted[0]['fixture_only'] = True if case == 'fixture' else 'false'
                        elif case == 'id':
                            corrupted[0]['id'] = 'changed'
                        else:
                            corrupted[0]['messages'][0]['content'] = '<audio>Different instruction.'
                        write_rows(trial_metadata, corrupted)
                        rejects('mock_boundary_wrong_' + case, lambda: validate_data_boundary(trial_args))
                    write_rows(trial_metadata, expected)
                for case, overrides in (('wrong_epochs', dict(max_epochs=4)),
                                        ('wrong_min_epochs', dict(min_epochs=4)),
                                        ('smoke', dict(smoke_steps=2)),
                                        ('implicit_scope', dict(data_scope='interface'))):
                    invalid = argparse.Namespace(**(vars(trial_args) | overrides))
                    rejects(scope + '_' + case, lambda: validate_data_boundary(invalid))
            legacy = [row for partition, count in (('train', 24), ('val', 8))
                      for row in [r for r in expected if r['split'] == partition][:count]]
            write_rows(trial_metadata, legacy)
            rejects(scope + '_legacy_24_8', lambda: validate_data_boundary(trial_args))
        interface = argparse.Namespace(metadata=metadata, smoke_steps=2, min_epochs=3, max_epochs=10)
        assert validate_data_boundary(interface)[3]['approval_chain_verified'] is False
        checks.append(dict(case='default_interface_two_updates', passed=True, formal_reuse_forbidden=True))
        interface.smoke_steps = 0
        rejects('no_explicit_scope_formal', lambda: validate_data_boundary(interface))
        broadcasts = []
        def broadcast(value, src):
            assert src == 0
            broadcasts.append(copy.deepcopy(value))
        with mock.patch.object(dist, 'broadcast_object_list', side_effect=broadcast):
            rejects('rank0_error_broadcast', lambda: main_process_result(
                lambda: (_ for _ in ()).throw(ValueError('synthetic preflight refusal')), 0, 2))
        with mock.patch.object(dist, 'broadcast_object_list', side_effect=lambda value, src: value.__setitem__(0, broadcasts[0][0])):
            rejects('rank1_same_error_refusal', lambda: main_process_result(lambda: None, 1, 2))
        assert len(broadcasts) == 1 and 'synthetic preflight refusal' in broadcasts[0][0]['error']
        model_fixture = root / 'synthetic_model_identity_only'
        model_fixture.mkdir()
        (model_fixture / 'model.safetensors').write_bytes(b'synthetic mock, never model weights')
        identity_args = argparse.Namespace(metadata=metadata, model=model_fixture, model_id='synthetic-mock',
            out=root / 'not-created', resume=None, command='train', data_scope='reviewed_pilot')
        first_payload, first_fingerprint = provenance(identity_args, rows, 1, data)
        changed_data = copy.deepcopy(data)
        changed_data['sidecar']['sha256'] = '0' * 64
        _, changed_fingerprint = provenance(identity_args, rows, 1, changed_data)
        assert first_fingerprint != changed_fingerprint and first_payload['data_provenance'] == data
        checks.append(dict(case='approval_identity_changes_fingerprint', passed=True, synthetic_mock=True))
        checkpoint = root / 'synthetic_checkpoint_marker_only'
        checkpoint.mkdir()
        atomic_json(checkpoint / 'complete.json', dict(schema=SCHEMA, fingerprint=first_fingerprint))
        rejects('resume_cross_approval_identity', lambda: load_checkpoint(
            checkpoint, None, None, changed_fingerprint, torch.device('cpu'), 0, 1))
        rejects('legacy_v4_identity_not_relabelled', lambda: load_checkpoint(
            checkpoint, None, None, 'legacy-v4-fingerprint', torch.device('cpu'), 0, 1))
        resume_out = root / 'synthetic_resume_preflight_only'
        resume_out.mkdir()
        resume_checkpoint = resume_out / 'checkpoint'
        resume_checkpoint.mkdir()
        resume_args = argparse.Namespace(metadata=metadata, model=model_fixture, model_id='synthetic-mock',
            out=resume_out, resume=resume_checkpoint, command='train', data_scope='interface',
            smoke_steps=2, min_epochs=3, max_epochs=10, seed=42, zero_stage1=False)
        real_rows, _, _, interface_data = validate_data_boundary(resume_args)
        resume_payload, resume_fingerprint = provenance(resume_args, real_rows, 1, interface_data)
        atomic_json(resume_out / 'provenance.json', resume_payload)
        for case, marker in (
                ('fingerprint', dict(schema=SCHEMA, fingerprint='different-approval-identity')),
                ('schema', dict(schema='synthetic-legacy-v4', fingerprint=resume_fingerprint))):
            atomic_json(resume_checkpoint / 'complete.json', marker)
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.dict(os.environ, WORLD_SIZE='1', RANK='0', LOCAL_RANK='0'))
                stack.enter_context(mock.patch.object(torch.cuda, 'is_available', return_value=True))
                stack.enter_context(mock.patch.object(torch.cuda, 'set_device'))
                stack.enter_context(mock.patch.object(torch.cuda, 'is_bf16_supported', return_value=True))
                stack.enter_context(mock.patch.object(torch, 'manual_seed'))
                loading = mock.Mock(side_effect=AssertionError('Model loading must not be reached'))
                boundary = mock.Mock(wraps=validate_data_boundary)
                stack.enter_context(mock.patch.dict(globals(), identify_family=loading, validate_data_boundary=boundary))
                try:
                    train(resume_args)
                except ValueError as error:
                    assert 'Resume checkpoint provenance/approval identity changed before model loading' in str(error)
                else:
                    raise AssertionError('Actual train preflight accepted a changed checkpoint identity')
                boundary.assert_called_once_with(resume_args)
                loading.assert_not_called()
            checks.append(dict(case='actual_train_preflight_resume_wrong_' + case, rejected=True,
                               real_preflight_called=True, cuda_capabilities_mocked=True,
                               pretrained_model_loaded=False, synthetic_mock=True))
    result = dict(event='data_selftest_passed', complete=True, cpu_only=True,
        synthetic_cpu_contract_fixture=True, boundary_tests_use_mock=True, actual_approval=False,
        formal_training_performed=False, pretrained_model_loaded=False,
        source_sha256=file_sha(__file__), helper_sha256=file_sha(h3_video_sft.__file__),
        validator_sha256=file_sha(fusion.__file__), checks=checks)
    if args.out:
        atomic_json(args.out, result)
    log(result)


def full_data_selftest(args):
    """Synthetic CPU scope/identity tests; this is not admission of any real dataset."""
    import sys
    import types
    import wave
    from unittest import mock
    torch.set_num_threads(2)
    if args.out and (not args.out.parent.is_dir() or args.out.exists()):
        raise ValueError('Full data selftest output requires a new JSON file in an existing parent')
    checks = []
    def rejects(name, operation):
        try:
            operation()
        except (ValueError, KeyError, TypeError, FileNotFoundError):
            checks.append(dict(case=name, rejected=True))
        else:
            raise AssertionError(f'Accepted invalid full boundary: {name}')
    with tempfile.TemporaryDirectory(prefix='omni-full-cpu-contract-') as directory:
        root = Path(directory)
        audio = {}
        for split in ('train', 'val'):
            audio[split] = root / (split + '.wav')
            with wave.open(str(audio[split]), 'wb') as stream:
                stream.setparams((1, 2, 16000, 81000, 'NONE', 'not compressed'))
                stream.writeframes(np.full(81000, 1 if split == 'train' else 2, dtype='<i2').tobytes())
        rows = [dict(id=f'{split}-{index:05d}', source_group=f'{split}-{index:05d}', split=split,
                     caption_version='synthetic-full-version', audios=[str(audio[split])],
                     audio_window_seconds=[0., 5.0625],
                     messages=[dict(role='user', content='<audio>Plan continuous dancing.'),
                               dict(role='assistant', content='The dancer steps sideways and turns smoothly.')])
                for split, size in (('train', 6000), ('val', 500)) for index in range(size)]
        metadata = root / 'omni.jsonl'
        def write_rows(value):
            metadata.write_text(''.join(json.dumps(row) + '\n' for row in value))
        write_rows(rows)
        boundary = argparse.Namespace(metadata=metadata, data_scope='source_label_full', smoke_steps=0,
                                      min_epochs=3, max_epochs=3)
        mock_validator = mock.Mock(return_value=dict(protocol='synthetic-mock-only',
            approval_chain_verified=False, human_review_completed=False, semantic_approval=False,
            all_verified_sources=[dict(path='synthetic-source', sha256='a' * 64, size_bytes=1)]))
        module = types.ModuleType('omni_source_label_full')
        module.source_label_full_provenance = mock_validator
        with mock.patch.dict(sys.modules, omni_source_label_full=module):
            loaded_rows, training_rows, val_rows, data = validate_data_boundary(boundary)
            assert len(training_rows) == 6000 and len(val_rows) == 500
            assert data['data_scope'] == 'source_label_full' and data['approval_chain_verified'] is False
            mock_validator.assert_called_once_with(metadata, 'synthetic-full-version',
                                                  loaded_rows, training_rows, val_rows)
            checks.append(dict(case='full_exact_boundary_enabled_6000_500', passed=True,
                               real_metadata_reader=True, validator_mocked=True))
            for name, overrides in (('smoke', dict(smoke_steps=2)), ('min_epochs', dict(min_epochs=4)),
                                    ('max_epochs', dict(max_epochs=10)), ('missing_scope', dict(data_scope='interface')),
                                    ('pilot_upgrade', dict(data_scope='source_label_pilot'))):
                invalid = argparse.Namespace(**{**vars(boundary), **overrides})
                rejects(name, lambda invalid=invalid: validate_data_boundary(invalid))
            for name, altered in (('tiny_upgrade', [rows[0], rows[-1]]),
                                  ('wrong_counts', rows[:-1]),
                                  ('multiple_versions', [{**rows[0], 'caption_version': 'other'}, *rows[1:]]),
                                  ('missing_version', [{k: v for k, v in rows[0].items() if k != 'caption_version'}, *rows[1:]]),
                                  ('fixture', [{**rows[0], 'fixture_only': True}, *rows[1:]]),
                                  ('false_string', [{**rows[0], 'fixture_only': 'false'}, *rows[1:]]),
                                  ('forbidden_reuse', [{**rows[0], 'formal_reuse_forbidden': True}, *rows[1:]])):
                write_rows(altered)
                rejects(name, lambda: validate_data_boundary(boundary))
            write_rows(rows)
        model = root / 'local-synthetic-model'
        model.mkdir()
        (model / 'fixture.bin').write_bytes(b'synthetic mock only, not a model')
        identity = argparse.Namespace(**{**vars(boundary), 'model': model, 'model_id': 'synthetic-only',
            'out': root / 'not-created', 'resume': None, 'command': 'train'})
        first_payload, first = provenance(identity, loaded_rows, 1, data)
        altered = copy.deepcopy(data)
        altered['all_verified_sources'][0]['sha256'] = 'b' * 64
        _, second = provenance(identity, loaded_rows, 1, altered)
        assert first != second and first_payload['source_sha256'] == file_sha(__file__)
        checks.append(dict(case='full_all_source_identity_changes_fingerprint', passed=True))
        checkpoint = root / 'bad-resume'
        checkpoint.mkdir()
        atomic_json(checkpoint / 'complete.json', dict(schema=SCHEMA, fingerprint=first))
        rejects('full_changed_sources_resume', lambda: load_checkpoint(
            checkpoint, None, None, second, torch.device('cpu'), 0, 1))
        rejects('full_legacy_pilot_resume', lambda: load_checkpoint(
            checkpoint, None, None, 'legacy-pilot-fingerprint', torch.device('cpu'), 0, 1))
        shared_error = []
        def broadcast(shared, src):
            if shared[0] is not None:
                shared_error[:] = copy.deepcopy(shared)
            else:
                shared[:] = copy.deepcopy(shared_error)
        with mock.patch.object(dist, 'broadcast_object_list', side_effect=broadcast):
            rejects('rank0_validation_error_broadcast', lambda: main_process_result(
                lambda: (_ for _ in ()).throw(ValueError('synthetic admission failure')), 0, 2))
            rejects('rank1_receives_validation_error', lambda: main_process_result(lambda: None, 1, 2))
        state = dict(updates=0, epoch=0, position=0)
        (root / 'checkpoint-u0000000-e00-p0000000').mkdir()
        with mock.patch.object(dist, 'broadcast_object_list', side_effect=broadcast), \
                mock.patch.object(dist, 'gather_object'):
            rejects('rank0_checkpoint_error_broadcast', lambda: save_checkpoint(
                root, None, None, state, first, torch.device('cpu'), 0, 2))
            rejects('rank1_receives_checkpoint_error', lambda: save_checkpoint(
                root, None, None, state, first, torch.device('cpu'), 1, 2))
    result = dict(event='full_data_selftest_passed', complete=True, cpu_only=True,
                  synthetic_cpu_contract_fixture=True, boundary_tests_use_mock=True,
                  actual_admission=False, formal_training_performed=False, pretrained_model_loaded=False,
                  source_sha256=file_sha(__file__), checks=checks)
    if args.out:
        atomic_json(args.out, result)
    log(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    subparsers.add_parser('selftest', help='CPU-only explicit fixtures; never loads pretrained weights')
    data_check = subparsers.add_parser('data-selftest', help='CPU synthetic/mock approval gate tests; never actual approval')
    data_check.add_argument('--out', type=Path)
    full_check = subparsers.add_parser('full-data-selftest', help='CPU mock full-scope boundary and source identity tests, not real admission')
    full_check.add_argument('--out', type=Path)
    processor_check = subparsers.add_parser('processor-test', help='CPU real local processor and real audio; no model weights')
    processor_check.add_argument('--model', type=Path, required=True)
    processor_check.add_argument('--audio', type=Path, required=True)
    processor_check.add_argument('--window', nargs=2, type=float, default=[0., 5.0625])
    processor_check.add_argument('--out', type=Path, required=True)
    training = subparsers.add_parser('train')
    predicting = subparsers.add_parser('infer')
    for item in (training, predicting):
        item.add_argument('--metadata', type=Path, required=True)
        item.add_argument('--model', type=Path, required=True)
        item.add_argument('--model-id', required=True, help='Explicit official/revision provenance, not a folder nickname')
        item.add_argument('--out', type=Path, required=True)
        item.add_argument('--attention', choices=['sdpa', 'flash_attention_2', 'eager'], default='sdpa')
        item.add_argument('--max-length', type=int, default=8192)
        item.add_argument('--max-new-tokens', type=int, default=1536)
    training.add_argument('--resume', type=Path)
    training.add_argument('--smoke-steps', type=int, default=0)
    training.add_argument('--data-scope', choices=['interface', 'reviewed_pilot', 'reviewed_full', 'source_label_pilot', 'source_label_full'], default='interface',
                          help='interface: two-update smoke; reviewed_pilot: 512/128 three epochs; reviewed_full: 6000/500 3..10 epochs; source_label_pilot: authorized unreviewed 512/128 three epochs; source_label_full: explicitly admitted authorized weak labels, exact 6000/500 three epochs')
    training.add_argument('--infer', action='store_true', help='Generate up to three val captions after checkpoint reload, including smoke')
    training.add_argument('--lora-rank', type=int, default=32)
    training.add_argument('--lora-alpha', type=int, default=64)
    training.add_argument('--lora-lr', type=float, default=5e-5)
    training.add_argument('--projector-lr', type=float, default=1e-5)
    training.add_argument('--accumulation', type=int, default=1, help='Local batch is exactly one; effective batch world*accumulation')
    training.add_argument('--grad-clip', type=float, default=1.)
    training.add_argument('--min-epochs', type=int, default=3)
    training.add_argument('--max-epochs', type=int, default=10)
    training.add_argument('--patience', type=int, default=3)
    training.add_argument('--relative-delta', type=float, default=.005)
    training.add_argument('--val-generate', type=int, choices=[1, 2, 3], default=3)
    training.add_argument('--save-every', type=int, default=100)
    training.add_argument('--seed', type=int, default=42)
    training.add_argument('--zero-stage1', action='store_true')
    training.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    predicting.add_argument('--checkpoint', type=Path)
    predicting.add_argument('--ids', nargs='*', default=[])
    args = parser.parse_args()
    if args.command == 'selftest':
        selftest()
    elif args.command == 'data-selftest':
        data_selftest(args)
    elif args.command == 'full-data-selftest':
        full_data_selftest(args)
    elif args.command == 'processor-test':
        processor_test(args)
    elif args.command == 'train':
        if not (3 <= args.min_epochs <= args.max_epochs <= 10 and args.patience == 3 and
                args.relative_delta == .005 and args.accumulation >= 1 and args.lora_rank > 0 and
                args.lora_alpha > 0 and args.lora_lr > 0 and args.projector_lr > 0 and
                args.grad_clip > 0 and args.save_every >= 0 and args.smoke_steps in (0, 2)):
            parser.error('Require 3..10 full epochs, patience3/delta.005, positive optimizer settings, smoke 0 or 2')
        train(args)
    else:
        inference(args)


if __name__ == '__main__':
    main()
