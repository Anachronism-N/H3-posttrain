#!/usr/bin/env python3
"""First-pass E2 comparison: run each usable baseline on the same val cases.

Generates one caption per case with each model, then reports the sol61 audit
metrics per model so the SFT captioner can be placed on the same scale. This is
an exploration harness; the frozen ChoreoBench protocol still owns the正式 numbers.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

OSS = Path('/data/oss_bucket_0/niezichao.nzc/m2d_train_v1')
RUNS = OSS / 'runs'
RUN_ID = os.environ.get('E2_RUN_ID') or time.strftime('e2_baselines_%Y%m%d_%H%M%S')
CASES = int(os.environ.get('E2_CASES', '40'))
MAX_NEW = int(os.environ.get('E2_MAX_NEW', '512'))
PROMPT = ("Listen to this music excerpt and describe one plausible solo dance phrase: "
          "the actions in order and at the same time, with force and transitions. "
          "No style labels, no timestamps.")

MODELS = [
    ('Qwen3-Omni-30B-A3B-Instruct', 'qwen3', None),
    ('Qwen2.5-Omni-7B', 'qwen', 'Qwen2_5OmniForConditionalGeneration'),
    ('MOSS-Music-8B-Instruct', 'moss', None),
    ('music-flamingo-2601-hf', 'flamingo', 'MusicFlamingoForConditionalGeneration'),
    ('audio-flamingo-3-hf', 'flamingo', 'AudioFlamingo3ForConditionalGeneration'),
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
        model = MossMusicModel.from_pretrained(str(path), config=config, dtype=torch.bfloat16,
                                              local_files_only=True, device_map='auto')
    else:
        import transformers
        cls = getattr(transformers, cls_name) if cls_name else transformers.Qwen3OmniMoeThinkerForConditionalGeneration
        model = cls.from_pretrained(str(path), dtype=torch.bfloat16, **kwargs)
    return model.eval(), processor


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
    messages = [dict(role='user', content=[dict(type='audio', audio=str(wav_path)),
                                           dict(type='text', text=PROMPT)])]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    batch = processor(text=text, audio=audio, sampling_rate=sr, return_tensors='pt')
    device = next(model.parameters()).device
    batch = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items()}
    batch = {k: (v.bfloat16() if torch.is_tensor(v) and v.is_floating_point() else v)
             for k, v in batch.items()}
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=MAX_NEW, do_sample=False)
    if isinstance(out, tuple):
        out = out[0]
    decoded = processor.batch_decode(out[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)
    if isinstance(decoded, tuple):
        decoded = decoded[0]
    return (decoded[0] if decoded else '').strip()


def main():
    data = Path('/checkpoint/binary') / RUN_ID / 'data'
    rows = [json.loads(l) for l in (data / 'val.jsonl').open() if l.strip()]
    rows.sort(key=lambda r: r['id'])
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
