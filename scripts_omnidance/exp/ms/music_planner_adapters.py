"""Local native-audio music planners with identical training/generation prefixes."""
import hashlib
from functools import wraps
import importlib.util
import inspect
import json
import math
from pathlib import Path
import re
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

FAMILIES = ('moss', 'music_flamingo', 'omni_captioner')
TASK_INSTRUCTION = (
    'Listen to the music and describe a dance choreography in two time windows: '
    '[0-2s] and [2-5.062s]. Describe grounded, varied body movements for each window. '
    'Output only the two time-marked choreography descriptions.'
)


def _local_moss_class(path, module_name, class_name):
    path = Path(path).resolve()
    package = '_music_planner_moss_' + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(path)]
        module.__package__ = package
        sys.modules[package] = module
    name = package + '.' + module_name
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path / (module_name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return getattr(sys.modules[name], class_name)


def load_native_processor(family, model_path):
    if family not in FAMILIES:
        raise ValueError(f'Unknown planner family: {family}')
    if family == 'moss':
        from transformers import AutoConfig
        config_cls = _local_moss_class(model_path, 'configuration_moss_music', 'MossMusicConfig')
        AutoConfig.register('moss_music', config_cls, exist_ok=True)
        cls = _local_moss_class(model_path, 'processing_moss_music', 'MossMusicProcessor')
        return cls.from_pretrained(model_path, local_files_only=True, enable_time_marker=True)
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model_path, local_files_only=True)


def _moss_whisper_layer_forward(self, *args, **kwargs):
    output = self._planner_native_forward(*args, **kwargs)
    return (output,) if isinstance(output, torch.Tensor) else output


def _moss_forward(self, *args, **kwargs):
    bound = inspect.signature(self._planner_native_forward).bind(*args, **kwargs)
    embeds = bound.arguments.get('inputs_embeds')
    if embeds is None:
        embeds = self.get_input_embeddings()(bound.arguments.get('input_ids'))
    device_type = embeds.device.type
    if torch.is_autocast_enabled(device_type):
        embeds = embeds.to(torch.get_autocast_dtype(device_type))
    bound.arguments['inputs_embeds'] = embeds
    return self._planner_native_forward(*bound.args, **bound.kwargs)


def _moss_register_deepstack(self, audio_input_mask, deepstack_audio_embeds):
    handles = self._planner_native_register_deepstack(audio_input_mask, deepstack_audio_embeds)
    handle_ids = {handle.id for handle in handles}
    for index, layer in enumerate(self.language_model.layers):
        for key, hook in layer._forward_hooks.items():
            if key in handle_ids:
                hook._planner_deepstack = True
                hook._planner_audio_embeds = deepstack_audio_embeds[index]
                hook._planner_audio_mask = audio_input_mask
                hook._planner_apply_deepstack = self._apply_deepstack_to_hidden_states
    return handles


def _moss_checkpoint_with_deepstack(layer, checkpoint):
    def run(function, *args, **kwargs):
        hooks = {key: hook for key, hook in layer._forward_hooks.items()
                 if getattr(hook, '_planner_deepstack', False)}
        if not hooks:
            return checkpoint(function, *args, **kwargs)
        def replay(*inputs, **options):
            existing = {key: layer._forward_hooks.get(key) for key in hooks}
            for (key, hook), embeds in zip(hooks.items(), inputs[len(args):]):
                def inject(module, incoming, output, hook=hook, embeds=embeds):
                    hidden = output[0] if isinstance(output, (tuple, list)) else output
                    hidden = hook._planner_apply_deepstack(hidden, hook._planner_audio_mask, embeds)
                    return (hidden,) + tuple(output[1:]) if isinstance(output, (tuple, list)) else hidden
                layer._forward_hooks[key] = inject
            try:
                return function(*inputs[:len(args)], **options)
            finally:
                for key, old in existing.items():
                    if old is None:
                        layer._forward_hooks.pop(key, None)
                    else:
                        layer._forward_hooks[key] = old
        embeds = tuple(hook._planner_audio_embeds for hook in hooks.values())
        return checkpoint(replay, *args, *embeds, **kwargs)
    return run


def _checkpoint_with_input_grad(checkpoint):
    def run(function, *args, **kwargs):
        # A frozen audio stem does not require gradients, but reentrant checkpoint
        # needs a grad-carrying tensor input to train the later audio LoRA layers.
        if args and isinstance(args[0], torch.Tensor) and not args[0].requires_grad:
            args[0].requires_grad_(True)
        return checkpoint(function, *args, **kwargs)
    return run


def _planner_enable_gradient_checkpointing(self, gradient_checkpointing_kwargs=None):
    result = self._planner_native_enable_checkpointing(
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    if (gradient_checkpointing_kwargs or {}).get('use_reentrant', True):
        self.enable_input_require_grads()
        for layer in self.modules():
            if getattr(layer, 'gradient_checkpointing', False):
                layer._gradient_checkpointing_func = _checkpoint_with_input_grad(
                    layer._gradient_checkpointing_func)
    if self._planner_family == 'moss':
        for layer in self.language_model.layers:
            layer._gradient_checkpointing_func = _moss_checkpoint_with_deepstack(
                layer, layer._gradient_checkpointing_func)
    return result


def _prepare_planner_checkpointing(model, family):
    if not hasattr(model, '_planner_native_enable_checkpointing'):
        model._planner_family = family
        model._planner_native_enable_checkpointing = model.gradient_checkpointing_enable
        model.gradient_checkpointing_enable = types.MethodType(
            _planner_enable_gradient_checkpointing, model)


def _prepare_moss_runtime(model):
    if not hasattr(model, '_planner_native_forward'):
        model._planner_native_forward = model.forward
        @wraps(type(model).forward)
        def forward(self, *args, **kwargs):
            return _moss_forward(self, *args, **kwargs)
        model.forward = types.MethodType(forward, model)
    # The released MOSS encoder indexes layer_outputs[0], while HF5 returns a Tensor.
    model.audio_encoder.config._attn_implementation = 'eager'
    for layer in model.audio_encoder.layers:
        if not hasattr(layer, '_planner_native_forward'):
            layer._planner_native_forward = layer.forward
            layer.forward = types.MethodType(_moss_whisper_layer_forward, layer)
    if not hasattr(model, '_planner_native_register_deepstack'):
        model._planner_native_register_deepstack = model._register_llm_deepstack_hooks
        model._register_llm_deepstack_hooks = types.MethodType(_moss_register_deepstack, model)
        # Native MOSS removes its per-forward DeepStack hooks before backward. Capture
        # only these hooks in each checkpoint closure so recomputation is identical.
    _prepare_planner_checkpointing(model, 'moss')


class PlannerAdapter:
    def __init__(self, family, model, processor, max_length=8192):
        if family not in FAMILIES:
            raise ValueError(f'Unknown planner family: {family}')
        if type(max_length) is not int or max_length <= 0:
            raise ValueError('max_length must be a positive integer')
        self.family, self.model, self.processor = family, model, processor
        self.max_length = max_length
        self._audio_identities = {}
        self.trainable_info = {}
        if family == 'moss' and hasattr(model, 'audio_encoder'):
            _prepare_moss_runtime(model)
        elif hasattr(model, 'gradient_checkpointing_enable'):
            _prepare_planner_checkpointing(model, family)

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    def _audio(self, row, silence=False):
        import soundfile as sf
        from scipy.signal import resample_poly
        path = Path(row['audio_path']).expanduser().resolve()
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
        expected = row.get('audio_sha256')
        if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('A declared audio_sha256 is required')
        if self._audio_identities.get(str(path)) != (identity, expected):
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError('Audio bytes differ from declared audio_sha256')
            self._audio_identities[str(path)] = (identity, expected)
        window = row.get('audio_window_seconds', [0., 5.0625])
        if window != [0., 5.0625]:
            raise ValueError('Only one full [0,5.0625] audio, never a physical window crop')
        audio, rate = sf.read(path, dtype='float32', always_2d=True)
        if audio.shape[1] != 1 or not np.isfinite(audio).all():
            raise ValueError('Require the original finite mono waveform')
        extra_frames = len(audio) - 5.0625 * rate
        # Keep the full normalized 24kHz source, including its observed five-frame overrun.
        normalized_source_overrun = rate == 24000 and 0 < extra_frames <= 5
        if abs(len(audio) / rate - 5.0625) > 1 / rate and not normalized_source_overrun:
            raise ValueError('Audio must contain the complete 5.0625-second source: '
                             f'id={row.get("id")}, rate={rate}, samples={len(audio)}, '
                             f'expected={5.0625 * rate}, extra_frames={extra_frames}')
        audio = audio[:, 0]
        if rate != 16000:
            divisor = math.gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
        if silence:
            audio = np.zeros_like(audio)
        return audio

    def _prompt(self):
        if self.family == 'moss':
            # Native MOSS default chat wrapper; the audio span is expanded by its processor.
            return self.processor._build_default_prompt(TASK_INSTRUCTION, has_audio=True)
        content = [{'type': 'audio', 'audio': 'native-waveform'}]
        if self.family != 'omni_captioner':
            content.append({'type': 'text', 'text': TASK_INSTRUCTION})
        return self.processor.apply_chat_template(
            [{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)

    def prefix(self, row, silence=False):
        audio = self._audio(row, silence=silence)
        self._last_audio_info = dict(processor_sample_rate=16000, n_samples=len(audio))
        prompt = self._prompt()
        if self.family == 'moss':
            batch = dict(self.processor(text=prompt, audios=[audio], return_tensors='pt'))
            batch['audio_input_mask'] = batch['input_ids'].eq(self.processor.audio_token_id)
        elif self.family == 'music_flamingo':
            # Keep native 30s audio padding for its fixed positional embedding;
            # text padding must not override the processor's audio kwargs.
            batch = dict(self.processor(text=[prompt], audio=[audio], sampling_rate=16000,
                                        return_tensors='pt', text_kwargs={'padding': False},
                                        audio_kwargs={'padding': 'max_length'}))
        else:
            batch = dict(self.processor(text=[prompt], audio=[audio], sampling_rate=16000,
                                        return_tensors='pt', padding=False))
        batch.pop('labels', None)
        if any('pixel_values' in key or 'grid_thw' in key for key in batch):
            raise ValueError('Visual features are forbidden in native audio-only planner inputs')
        if batch['input_ids'].shape[0] != 1:
            raise ValueError('Native processor did not produce exactly one source')
        if batch['input_ids'].shape[1] > self.max_length:
            raise ValueError('Expanded audio prefix exceeds max_length; truncation forbidden')
        return batch

    def eos_token_ids(self):
        generation = getattr(self.model, 'generation_config', None)
        eos = getattr(generation, 'eos_token_id', None)
        if eos is None:
            eos = self.tokenizer.eos_token_id
        ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        if not ids or any(type(i) is not int for i in ids):
            raise ValueError('Native tokenizer/model has no valid EOS')
        return ids

    def _assistant_ids(self, text):
        if not isinstance(text, str) or not text.strip() or '<|' in text or '<audio>' in text:
            raise ValueError('Assistant target must be nonempty text without reserved chat tokens')
        target = self.tokenizer.encode(text, add_special_tokens=False)
        # All three native Qwen chat templates terminate an assistant turn with im_end.
        terminator = self.tokenizer.encode('<|im_end|>', add_special_tokens=False)
        if len(terminator) != 1 or terminator[0] not in self.eos_token_ids():
            raise ValueError('Native assistant turn terminator is not a single generation EOS')
        return torch.tensor([target + terminator], dtype=torch.long)

    def collate(self, rows):
        if len(rows) != 1:
            raise ValueError('Native audio feature packing supports microbatch=1 only')
        row = rows[0]
        batch = self.prefix(row)
        prefix_ids = batch['input_ids']
        target = self._assistant_ids(row['target_text'])
        if prefix_ids.shape[1] + target.shape[1] > self.max_length:
            raise ValueError('Full prefix plus assistant target exceeds max_length; truncation forbidden')
        batch['input_ids'] = torch.cat((prefix_ids, target), dim=1)
        batch['attention_mask'] = torch.cat((batch['attention_mask'], torch.ones_like(target)), dim=1)
        if 'audio_input_mask' in batch:
            batch['audio_input_mask'] = torch.cat((batch['audio_input_mask'], torch.zeros_like(target, dtype=torch.bool)), dim=1)
        for key in ('token_type_ids', 'mm_token_type_ids'):
            if key in batch:
                batch[key] = torch.cat((batch[key], torch.zeros_like(target)), dim=1)
        batch['labels'] = torch.cat((torch.full_like(prefix_ids, -100), target.clone()), dim=1)
        if not torch.equal(batch['input_ids'][:, :prefix_ids.shape[1]], prefix_ids):
            raise ValueError('Training prefix differs from native inference prefix')
        return batch

    def _move(self, batch):
        parameter = next(self.model.parameters())
        return {key: value.to(device=parameter.device,
                             dtype=parameter.dtype if value.is_floating_point() else value.dtype)
                if isinstance(value, torch.Tensor) else value
                for key, value in batch.items() if key != 'labels'}

    @torch.no_grad()
    def generate(self, row, max_new_tokens=2048, silence=False, seed=0):
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError('max_new_tokens must be positive')
        batch = self.prefix(row, silence=silence)
        prefix_ids = batch['input_ids']
        if prefix_ids.shape[1] + max_new_tokens > self.max_length:
            raise ValueError('Generation budget exceeds context; truncation forbidden')
        self.model.eval()
        # Greedy generation is seed-independent and never touches the reference target.
        parameter = next(self.model.parameters())
        with torch.autocast(parameter.device.type, dtype=torch.bfloat16,
                            enabled=parameter.dtype == torch.bfloat16):
            output = self.model.generate(**self._move(batch), do_sample=False,
                                         num_beams=1, repetition_penalty=1.0,
                                         no_repeat_ngram_size=0, encoder_repetition_penalty=1.0,
                                         max_new_tokens=max_new_tokens, use_cache=True)
        sequences = output.sequences if hasattr(output, 'sequences') else output
        if isinstance(sequences, tuple):
            sequences = sequences[0]
        if sequences.ndim != 2 or sequences.shape[0] != 1:
            raise ValueError('Native model returned an invalid text generation')
        prefix_length = prefix_ids.shape[1]
        if not torch.equal(sequences[:, :prefix_length].cpu(), prefix_ids.cpu()):
            raise ValueError('Native generation output does not contain the exact input prefix')
        new_ids = sequences[0, prefix_length:].cpu().tolist()
        eos = self.eos_token_ids()
        raw = self.tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        reason = 'eos' if new_ids and new_ids[-1] in eos else ('max_new_tokens' if len(new_ids) == max_new_tokens else 'unknown_stop')
        return dict(raw_text=raw, new_token_ids=new_ids, eos_token_ids=eos,
                    stop_reason=reason, native_stop=False,
                    **self._last_audio_info,
                    decoding=dict(do_sample=False, num_beams=1, repetition_penalty=1.0,
                                  no_repeat_ngram_size=0, encoder_repetition_penalty=1.0),
                    input_tokens=prefix_length, seed=seed,
                    silence_applied=bool(silence), audio_input='zero-waveform' if silence else 'full-waveform')

    @torch.no_grad()
    def teacher_forced(self, row):
        batch = self.collate([row])
        self.model.eval()
        parameter = next(self.model.parameters())
        with torch.autocast(parameter.device.type, dtype=torch.bfloat16,
                            enabled=parameter.dtype == torch.bfloat16):
            output = self.model(**self._move(batch))
        labels = batch['labels'][:, 1:].to(output.logits.device)
        logits = output.logits[:, :-1].float()
        count = int(labels.ne(-100).sum())
        total = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                ignore_index=-100, reduction='sum').item()
        return dict(sum=total, token_count=count, mean_nll=total / count)

    def save_pretrained(self, path, state_dict=None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path, state_dict=state_dict, safe_serialization=True)
        if hasattr(self.processor, 'save_pretrained'):
            self.processor.save_pretrained(path)
        else:
            import inspect
            import shutil
            self.tokenizer.save_pretrained(path)
            source = Path(inspect.getfile(type(self.processor))).parent
            for name in ('processing_moss_music.py', 'configuration_moss_music.py', 'preprocessor_config.json'):
                shutil.copyfile(source / name, path / name)
            base = self.model.get_base_model() if hasattr(self.model, 'get_base_model') else self.model
            base.config.save_pretrained(path)
        (path / 'planner_adapter.json').write_text(json.dumps(dict(
            family=self.family, max_length=self.max_length, trainable_info=self.trainable_info,
            prefix_policy='native-audio-only-no-target-in-prefix',
            time_markers=self.family == 'moss'), ensure_ascii=False, indent=2))


def _cast_parameter_dtype(parameter, dtype):
    parameter.data = parameter.data.to(dtype=dtype)
    # ZeRO3 can leave a BF16 parameter with FP32 partition storage after loading.
    # Mutate tensor storage rather than replacing the tensor and losing DS attributes.
    for attribute in ('ds_tensor', 'ds_secondary_tensor'):
        partition = getattr(parameter, attribute, None)
        if partition is not None:
            partition.data = partition.data.to(dtype=dtype)


class _PlannerDeepSpeedHookOwner:
    def __getattr__(self, name):
        # ZeRO installs child hooks before parent hooks. PEFT's attribute proxy
        # must not make a wrapper reuse a child's module-bound autograd closure.
        if name in ('pre_bwd_fn', 'post_bwd_fn', 'ds_grads_remaining',
                    'applied_pre_backward_ref_cnt'):
            return torch.nn.Module.__getattr__(self, name)
        return super().__getattr__(name)


_PLANNER_HOOK_OWNER_CLASSES = {}


def _isolate_deepspeed_hook_owners(model):
    for module in model.modules():
        cls = type(module)
        if not cls.__module__.startswith('peft.'):
            continue
        if cls not in _PLANNER_HOOK_OWNER_CLASSES:
            _PLANNER_HOOK_OWNER_CLASSES[cls] = type(
                '_Planner' + cls.__name__, (_PlannerDeepSpeedHookOwner, cls),
                {'__module__': __name__})
        module.__class__ = _PLANNER_HOOK_OWNER_CLASSES[cls]


def attach_lora(model, family, rank=32, alpha=64, audio_last_n=4):
    from peft import LoraConfig, get_peft_model
    if family not in FAMILIES or rank <= 0 or alpha <= 0 or audio_last_n <= 0:
        raise ValueError('Invalid family or LoRA scope')
    model.requires_grad_(False)
    if family == 'moss':
        _prepare_moss_runtime(model)
        audio_root, lm_root = 'audio_encoder', 'language_model'
        projectors = ['audio_adapter'] + [f'deepstack_audio_merger_list.{i}'
                      for i in range(len(model.deepstack_audio_merger_list))]
    elif family == 'music_flamingo':
        audio_root, lm_root = 'model.audio_tower', 'model.language_model'
        projectors = ['model.multi_modal_projector']
    else:
        audio_root, lm_root = 'audio_tower', 'model'
        projectors = ['audio_tower.proj1', 'audio_tower.proj2']
    _prepare_planner_checkpointing(model, family)
    audio_layers = len(model.get_submodule(audio_root).layers)
    if audio_layers < audio_last_n:
        raise ValueError('Native audio tower contains fewer layers than the requested scope')
    selected, experts = [], []
    lm_pattern = re.escape(lm_root) + r'[.]layers[.][0-9]+[.](self_attn|mlp)([.].+)?'
    audio_pattern = re.escape(audio_root) + r'[.]layers[.]([0-9]+)[.].+'
    for name, module in model.named_modules():
        audio_match = re.fullmatch(audio_pattern, name)
        if isinstance(module, torch.nn.Linear) and (
                re.fullmatch(lm_pattern, name) or
                (audio_match and int(audio_match[1]) >= audio_layers - audio_last_n)):
            selected.append(name)
    for name, parameter in model.named_parameters():
        if re.fullmatch(re.escape(lm_root) + r'[.]layers[.][0-9]+[.]mlp[.]experts[.](gate_up_proj|down_proj)', name):
            if len(getattr(parameter, 'ds_shape', parameter.shape)) != 3:
                raise ValueError(f'Invalid native fused-expert shape: {name}')
            experts.append(name)
    if not selected or not any(name.startswith(audio_root + '.') for name in selected):
        raise ValueError('No native language/audio LoRA targets were found')
    for name in projectors:
        model.get_submodule(name)
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0., bias='none',
                        target_modules=selected, target_parameters=experts or None,
                        modules_to_save=projectors)
    model = get_peft_model(model, config)
    _isolate_deepspeed_hook_owners(model)
    scope = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not ('lora_' in name or '.modules_to_save.' in name):
            raise ValueError(f'Unexpected unfrozen native base parameter: {name}')
        if any(token in name.lower() for token in ('visual', 'vision_tower', 'talker', 'code2wav', 'lm_head')):
            raise ValueError(f'Forbidden music-only trainable parameter: {name}')
        _cast_parameter_dtype(parameter, torch.float32)
        shape = list(getattr(parameter, 'ds_shape', parameter.shape))
        scope.append(dict(name=name, shape=shape, parameters=math.prod(shape),
                          dtype=str(parameter.dtype),
                          group='lora' if 'lora_' in name else 'projector',
                          scope='projector' if '.modules_to_save.' in name else
                                'audio_lora' if audio_root + '.' in name else 'language_lora'))
    if not any(item['scope'] == 'audio_lora' for item in scope):
        raise ValueError('Audio LoRA scope is empty after PEFT attachment')
    return model, dict(target_modules=selected, target_parameters=experts,
                       projectors=projectors, audio_root=audio_root, language_root=lm_root,
                       audio_last_layers=list(range(audio_layers - audio_last_n, audio_layers)),
                       rank=rank, alpha=alpha, trainable=scope)


def configure_trainable(adapter, rank=32, alpha=64, audio_last_n=4):
    adapter.model, adapter.trainable_info = attach_lora(
        adapter.model, adapter.family, rank=rank, alpha=alpha, audio_last_n=audio_last_n)
    return adapter.trainable_info


def load_planner_adapter(family, model_path, checkpoint=None, device=None,
                         dtype=torch.bfloat16, max_length=8192, lora_rank=None,
                         lora_alpha=64, audio_last_n=4):
    if family not in FAMILIES:
        raise ValueError(f'Unknown planner family: {family}')
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError('Only an existing local model directory is accepted')
    processor = load_native_processor(family, path)
    model_kwargs = {}
    if family == 'moss':
        from transformers import WhisperConfig
        cls = _local_moss_class(path, 'modeling_moss_music', 'MossMusicModel')
        config = cls.config_class.from_pretrained(path, local_files_only=True)
        # The released minimal config stores a dict, while its shipped encoder uses attributes.
        config.audio_config = WhisperConfig(**config.audio_config)
        model_kwargs['config'] = config
    elif family == 'music_flamingo':
        from transformers import MusicFlamingoForConditionalGeneration as cls
    else:
        # Load only the native thinker subtree: no talker/code2wav parameters or execution.
        from transformers import Qwen3OmniMoeThinkerForConditionalGeneration as cls
    model = cls.from_pretrained(path, local_files_only=True, dtype=dtype, **model_kwargs)
    for parameter in model.parameters():
        if parameter.is_floating_point():
            _cast_parameter_dtype(parameter, dtype)
    if device is not None:
        model = model.to(device)
    adapter = PlannerAdapter(family, model, processor, max_length=max_length)
    if lora_rank is not None:
        configure_trainable(adapter, rank=lora_rank, alpha=lora_alpha, audio_last_n=audio_last_n)
    if checkpoint is not None:
        from peft import PeftConfig, get_peft_model
        from peft.utils import load_peft_weights, set_peft_model_state_dict, get_peft_model_state_dict
        checkpoint = Path(checkpoint)
        export = checkpoint / 'adapter' if (checkpoint / 'adapter').is_dir() else checkpoint
        if adapter.model is not model:
            raise ValueError('Choose either an existing checkpoint or fresh LoRA configuration')
        metadata = json.loads((export / 'planner_adapter.json').read_text())
        if metadata['family'] != family:
            raise ValueError('Checkpoint planner family differs')
        config = PeftConfig.from_pretrained(export, local_files_only=True)
        config.inference_mode = False
        adapter.model = get_peft_model(model, config)
        _isolate_deepspeed_hook_owners(adapter.model)
        # PEFT's ordinary BF16-base load downcasts modules_to_save before upcasting LoRA.
        # Allocate every learned tensor in FP32 before copying the portable state instead.
        for parameter in adapter.model.parameters():
            if parameter.requires_grad:
                _cast_parameter_dtype(parameter, torch.float32)
        saved = load_peft_weights(export, device='cpu', local_files_only=True)
        result = set_peft_model_state_dict(adapter.model, saved)
        actual = get_peft_model_state_dict(adapter.model)
        if result.unexpected_keys or set(actual) != set(saved) or any(
                not torch.equal(actual[key].detach().cpu(), saved[key]) for key in saved):
            raise ValueError('Portable planner adapter state did not restore exactly')
        adapter.model.eval().requires_grad_(False)
        adapter.processor = load_native_processor(family, export)
        adapter.trainable_info = metadata.get('trainable_info', {})
    return adapter
