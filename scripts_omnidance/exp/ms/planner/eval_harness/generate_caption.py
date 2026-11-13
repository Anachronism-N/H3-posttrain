#!/usr/bin/env python3
"""Generate one dance caption for one 5.0625 s music window.

Loads the Qwen3-Omni-Captioner base model plus the published LoRA adapter and
runs the validated decoding configuration (temperature 0.7 / top-p 0.9 by
default; greedy decoding collapses onto one opener phrase).

Usage:
    python generate_caption.py --model /path/to/Qwen3-Omni-30B-A3B-Captioner-full \
        --adapter /path/to/adapter --audio window.wav [--temperature 0.7] [--seed 0]

The adapter was trained on 5.0625 s windows and enforces that length; cut the
window yourself rather than relying on padding.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATES = [HERE / 'scripts_omnidance/exp/ms', HERE.parents[1], HERE.parent,
              Path('/ssd/2/niezichao.nzc/sound-caption/h3-posttrain/H3-posttrain/scripts_omnidance/exp/ms')]
MS = next((p for p in CANDIDATES if (p / 'music_planner_adapters.py').is_file()), CANDIDATES[-1])
EXPECTED_SECONDS = 5.0625


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load(model, adapter, dtype='bfloat16'):
    import torch
    sys.path.insert(0, str(MS))
    from music_planner_adapters import load_planner_adapter
    return load_planner_adapter(family='omni_captioner', model_path=Path(model),
                                checkpoint=Path(adapter), device_map='auto',
                                dtype=getattr(torch, dtype))


def caption(planner, audio, temperature=0.7, top_p=0.9, seed=0, max_new_tokens=2048):
    import soundfile as sf
    import torch
    info = sf.info(str(audio))
    if abs(info.duration - EXPECTED_SECONDS) > 1e-3:
        raise ValueError('expected a %.4f s window, got %.4f s: cut the window first'
                         % (EXPECTED_SECONDS, info.duration))
    source = dict(id=Path(audio).stem, audio_path=str(audio),
                  audio_sha256=sha256_file(audio),
                  audio_window_seconds=[0.0, EXPECTED_SECONDS])
    batch = planner.prefix(source)
    planner.model.eval()
    torch.manual_seed(seed)
    parameter = next(planner.model.parameters())
    with torch.no_grad(), torch.autocast(parameter.device.type, dtype=torch.bfloat16,
                                         enabled=parameter.dtype == torch.bfloat16):
        if temperature and temperature > 0:
            output = planner.model.generate(**planner._move(batch), do_sample=True,
                                            temperature=temperature, top_p=top_p, num_beams=1,
                                            repetition_penalty=1.05, no_repeat_ngram_size=4,
                                            max_new_tokens=max_new_tokens, use_cache=True)
        else:
            output = planner.model.generate(**planner._move(batch), do_sample=False, num_beams=1,
                                            repetition_penalty=1.05, no_repeat_ngram_size=4,
                                            max_new_tokens=max_new_tokens, use_cache=True)
    sequences = output.sequences if hasattr(output, 'sequences') else output
    ids = sequences[0, batch['input_ids'].shape[1]:].cpu().tolist()
    return planner.tokenizer.decode(ids, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--adapter', required=True, type=Path)
    parser.add_argument('--audio', required=True, type=Path, action='append')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='0 disables sampling (not recommended: greedy collapses openers)')
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dtype', default='bfloat16', choices=('bfloat16', 'float32'))
    args = parser.parse_args()

    planner = load(args.model, args.adapter, args.dtype)
    results = []
    for audio in args.audio:
        text = caption(planner, audio, args.temperature, args.top_p, args.seed)
        results.append(dict(audio=str(audio), caption=text))
        print(json.dumps(results[-1], ensure_ascii=False))
    return results


if __name__ == '__main__':
    main()
