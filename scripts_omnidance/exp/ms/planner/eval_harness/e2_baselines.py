#!/usr/bin/env python3
"""First-pass E2 comparison: run each usable baseline on the same val cases.

Generates one caption per case with each model, then reports the sol61 audit
metrics per model so the SFT captioner can be placed on the same scale. This is
an exploration harness; the frozen ChoreoBench protocol still owns the正式 numbers.
"""
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

OSS = Path('/data/oss_bucket_0/niezichao.nzc/m2d_train_v1')
RUNS = OSS / 'runs'
RUN_ID = os.environ.get('E2_RUN_ID') or time.strftime('e2_baselines_%Y%m%d_%H%M%S')
CASES = int(os.environ.get('E2_CASES', '20'))
MAX_NEW = int(os.environ.get('E2_MAX_NEW', '512'))
PROMPT = ("Listen to this music excerpt and describe one plausible solo dance phrase: "
          "the actions in order and at the same time, with force and transitions. "
          "No style labels, no timestamps.")

MODELS = [
    ('MOSS-Music-8B-Instruct', 'moss', None),
]


def load_model(kind, cls_name, path):
    import torch
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
    kwargs = dict(local_files_only=True, trust_remote_code=True, device_map='auto')
    if kind == 'qwen3':
        # the generic AutoProcessor path hands back a text tokenizer without the
        # audio/image placeholders this chat template needs
        from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor
        processor = Qwen3OmniMoeProcessor.from_pretrained(str(path), local_files_only=True)
        import transformers
        model = transformers.Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            str(path), dtype=torch.bfloat16, local_files_only=True, device_map='auto')
        return model.eval(), processor
    if kind == 'moss':
        import importlib.util
        import types
        from transformers import WhisperConfig

        # the shipped modules use relative imports, so load them as a package
        package = types.ModuleType('mp_moss_pkg')
        package.__path__ = [str(path)]
        sys.modules['mp_moss_pkg'] = package

        def local(name, filename):
            full = 'mp_moss_pkg.' + name
            spec = importlib.util.spec_from_file_location(full, str(path / filename))
            module = importlib.util.module_from_spec(spec)
            sys.modules[full] = module
            spec.loader.exec_module(module)
            return module

        MossMusicConfig = local('configuration_moss_music', 'configuration_moss_music.py').MossMusicConfig
        MossMusicModel = local('modeling_moss_music', 'modeling_moss_music.py').MossMusicModel
        config = MossMusicConfig.from_pretrained(str(path), local_files_only=True)
        config.audio_config = WhisperConfig(**config.audio_config)
        patch_whisper_layer()
        # single device: auto-sharding splits the audio encoder away from the
        # embedding layer and masked_scatter_ then fails across devices
        model = MossMusicModel.from_pretrained(str(path), config=config, dtype=torch.bfloat16,
                                              local_files_only=True, device_map={'': 0})
        patch_moss_audio_reuse(MossMusicModel)
    else:
        import transformers
        cls = getattr(transformers, cls_name) if cls_name else transformers.Qwen3OmniMoeThinkerForConditionalGeneration
        model = cls.from_pretrained(str(path), dtype=torch.bfloat16, **kwargs)
    return model.eval(), processor




def patch_moss_audio_reuse(cls):
    """Drop audio inputs on decode steps that do not carry the audio span.

    transformers 5.x does not always hand `prepare_inputs_for_generation` a
    cache_position, so the model re-injects `audio_data` for the appended token
    and masked_scatter_ then sees [1, T+1] embeddings against a [1, T] mask.
    """
    import functools
    original_forward = cls.forward

    @functools.wraps(original_forward)
    def forward(self, *args, **kwargs):
        mask = kwargs.get('audio_input_mask')
        ids = kwargs.get('input_ids')
        if mask is not None and ids is not None and ids.shape[1] != mask.shape[1]:
            kwargs['audio_data'] = None
            kwargs['audio_data_seqlens'] = None
            kwargs['audio_input_mask'] = None
        return original_forward(self, *args, **kwargs)

    original_prepare = cls.prepare_inputs_for_generation

    @functools.wraps(original_prepare)
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        out = original_prepare(self, input_ids, past_key_values=past_key_values, **kwargs)
        mask = out.get('audio_input_mask')
        ids = out.get('input_ids')
        if mask is not None and ids is not None and ids.shape[1] != mask.shape[1]:
            out['audio_data'] = None
            out['audio_data_seqlens'] = None
            out['audio_input_mask'] = None
        return out

    cls.forward = forward
    cls.prepare_inputs_for_generation = prepare_inputs_for_generation


def patch_whisper_layer():
    """Version-agnostic WhisperEncoderLayer.forward.

    transformers 5.18 changed WhisperAttention to an attention-interface call, and
    the MOSS encoder feeds it tensors shaped for the older API. Re-implementing
    the layer against the module's own projections removes that coupling.
    """
    import inspect
    import torch
    import torch.nn.functional as F
    from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer

    print('[moss] transformers', __import__('transformers').__version__, flush=True)
    print('[moss] layer attrs', sorted(n for n, _ in WhisperEncoderLayer().named_children()) if False else
          [n for n in dir(WhisperEncoderLayer) if 'forward' in n], flush=True)

    original = WhisperEncoderLayer.forward
    try:
        print('[moss] original forward source:\n' + inspect.getsource(original), flush=True)
    except Exception as error:
        print('[moss] source unavailable: %r' % (error,), flush=True)

    def forward(self, hidden_states, attention_mask=None, layer_head_mask=None, output_attentions=False):
        attn = self.self_attn
        residual = hidden_states
        h = self.self_attn_layer_norm(hidden_states)
        batch, length, width = h.shape
        heads = int(getattr(attn, 'num_heads', getattr(attn, 'heads', width // getattr(attn, 'head_dim', 64))))
        dim = int(getattr(attn, 'head_dim', width // heads))
        q = attn.q_proj(h).reshape(batch, length, heads, dim).transpose(1, 2)
        k = attn.k_proj(h).reshape(batch, length, heads, dim).transpose(1, 2)
        v = attn.v_proj(h).reshape(batch, length, heads, dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (dim ** 0.5)
        if attention_mask is not None:
            mask = attention_mask
            while mask.dim() > scores.dim():
                mask = mask.squeeze(1)
            if mask.dim() == scores.dim() and mask.shape[0] in (1, batch):
                scores = scores + mask.to(scores.dtype)
        probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        out = torch.matmul(probs, v).transpose(1, 2).reshape(batch, length, width)
        hidden_states = residual + attn.out_proj(out)
        residual = hidden_states
        h = self.final_layer_norm(hidden_states)
        act = getattr(self, 'activation_fn', None) or getattr(self, 'activation_function')
        hidden_states = residual + self.fc2(act(self.fc1(h)))
        return (hidden_states,)

    WhisperEncoderLayer.forward = forward
    return original


def target_rate(processor):
    for holder in ('feature_extractor', 'audio_processor', 'processor'):
        inner = getattr(processor, holder, None)
        rate = getattr(inner, 'sampling_rate', None)
        if isinstance(rate, int) and rate > 0:
            return rate
    return 16000


def read_audio(path, rate):
    import numpy as np
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != rate:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(sr), int(rate))
        audio = resample_poly(audio, rate // g, int(sr) // g).astype('float32')
    return audio, int(rate)


def generate(model, processor, wav_path, kind):
    import torch
    audio, sr = read_audio(wav_path, target_rate(processor))
    if kind == 'moss':
        # MossMusicProcessor has no chat template; it expands the audio span itself
        text = processor._build_default_prompt(PROMPT, has_audio=True)
        batch = dict(processor(text=text, audios=[audio], return_tensors='pt'))
        batch['audio_input_mask'] = batch['input_ids'].eq(processor.audio_token_id)
        if not getattr(generate, 'logged', False):
            generate.logged = True
            print('[moss] batch', {k: (tuple(v.shape) if hasattr(v, 'shape') else type(v).__name__) for k, v in batch.items()}, flush=True)
            print('[moss] audio tokens', int(batch['audio_input_mask'].sum()), 'audio_token_id', processor.audio_token_id, flush=True)
    else:
        messages = [dict(role='user', content=[dict(type='audio', audio=str(wav_path)),
                                               dict(type='text', text=PROMPT)])]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        batch = processor(text=text, audio=audio, sampling_rate=sr, return_tensors='pt')
    device = next(model.parameters()).device
    batch = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items()}
    batch = {k: (v.bfloat16() if torch.is_tensor(v) and v.is_floating_point() else v)
             for k, v in batch.items()}
    if False and kind == 'moss' and not getattr(generate, 'probed', False):
        generate.probed = True
        try:
          with torch.no_grad():
            enc = model.audio_encoder(batch['audio_data'], batch['audio_data_seqlens']).last_hidden_state
            print('[moss] encoder out', tuple(enc.shape), 'nan', int(torch.isnan(enc).sum()),
                  'absmean', float(enc.float().abs().mean()), flush=True)
            probe = model(**batch)
            logits = probe.logits[0, -1].float()
            top = torch.topk(logits, 5)
            print('[moss] logits nan', int(torch.isnan(logits).sum()), 'min', float(logits.min()),
                  'max', float(logits.max()), flush=True)
            print('[moss] top5', [(int(i), processor.tokenizer.decode([int(i)])) for i in top.indices],
                  [round(float(v), 3) for v in top.values], flush=True)
            print('[moss] gen cfg eos', model.generation_config.eos_token_id,
                  'pad', model.generation_config.pad_token_id, flush=True)
        except Exception as error:
            print('[moss] probe failed: %r' % (error,), flush=True)
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=MAX_NEW, do_sample=False)
        if False and kind == 'moss' and not getattr(generate, 'raw', False):
            generate.raw = True
            ids = (out.sequences if hasattr(out, 'sequences') else out)
            print('[moss] raw out', tuple(ids.shape), ids[0, batch['input_ids'].shape[1]:][:16].tolist(), flush=True)
    if isinstance(out, tuple):
        out = out[0]
    decoded = processor.batch_decode(out[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)
    if isinstance(decoded, tuple):
        decoded = decoded[0]
    return (decoded[0] if decoded else '').strip()


def main():
    data = Path('/checkpoint/binary') / RUN_ID / 'data'
    rows = [json.loads(l) for l in (data / 'val.jsonl').open() if l.strip()]
    # same selection rule as eval_many.py so the cases match the checkpoint audits
    rows.sort(key=lambda r: hashlib.sha256(('many:' + r['id']).encode()).hexdigest())
    rows = rows[:CASES]
    out_dir = RUNS / RUN_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, kind, cls_name in MODELS:
        path = OSS / 'models' / name
        started = time.time()
        rec = dict(model=name, kind=kind)
        try:
            model, processor = load_model(kind, cls_name, path)
            rec['load_s'] = round(time.time() - started, 1)
            gen = []
            for index, row in enumerate(rows):
                wav = data / row['audio_path']
                try:
                    text = generate(model, processor, wav, kind)
                    gen.append(dict(id=row['id'], text=text, reference=row['target_text']))
                except Exception as error:
                    gen.append(dict(id=row['id'], text='', reference=row['target_text'],
                                    error=traceback.format_exc()[-500:]))
                if index % 10 == 9:
                    print('[%s] %d/%d' % (name, index + 1, len(rows)), flush=True)
            f = out_dir / ('%s.jsonl' % name.replace('/', '_'))
            f.write_text('\n'.join(json.dumps(g, ensure_ascii=False) for g in gen) + '\n')
            rec['generated'] = sum(1 for g in gen if g['text'])
            rec['empty'] = sum(1 for g in gen if not g['text'])
            rec['seconds'] = round(time.time() - started, 1)
        except Exception:
            rec['error'] = traceback.format_exc()[-1500:]
        summary[name] = rec
        print(json.dumps(rec, ensure_ascii=False)[:500], flush=True)
        (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        try:
            del model
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
