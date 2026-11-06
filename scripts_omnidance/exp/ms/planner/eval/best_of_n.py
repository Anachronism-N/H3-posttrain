#!/usr/bin/env python3
"""RSI-lite: sample several captions per audio and keep the best by a rule score."""
import argparse
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

MS = Path('/ssd/2/niezichao.nzc/artifacts/h3-jev-worktree-120fe6d/scripts_omnidance/exp/ms')
sys.path.insert(0, str(MS))

MARKER = re.compile(r'\[(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)\s*s?\]')
STOPS = ('\nassistant', '\nAssistant', 'Assistant:', 'End of interaction', '###', '<|im_end|>')
ACTION = re.compile(r'\b(?:wrist|elbow|knee|ankle|shoulder|hip|head|torso|up|down|left|right|increas|decreas|cross|extend|bend|turn|sway|reach|rais|lower)\w*', re.I)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def clean(text):
    cut = len(text)
    for token in STOPS:
        index = text.find(token)
        if index != -1:
            cut = min(cut, index)
    text = re.sub(r'\[\d+(?:\.\d+)?\s*[-–—]\s*\d+(?:\.\d+)?\s*s?\]\s*$', '', text[:cut])
    return text.strip()


def junk_ratio(text):
    lines = [line for line in text.splitlines() if line.strip()]
    short = sum(1 for line in lines if len(line.strip()) < 12)
    letters = sum(ch.isalpha() or ch.isspace() or ch in '.,;:[]-0123456789' for ch in text)
    return min(1.0, short / max(len(lines), 1) * 0.5 + (1 - letters / max(len(text), 1)))


def score(text):
    markers = MARKER.findall(text)
    cover = min(max((float(end) for _, end in markers), default=0.0), 5.0625) / 5.0625
    tokens = re.findall(r"[a-z]+(?:['’][a-z]+)?", text.lower())
    grams = [tuple(tokens[i:i + 8]) for i in range(max(0, len(tokens) - 7))]
    repetition = 1 - len(set(grams)) / len(grams) if grams else 1.0
    actions = len(set(ACTION.findall(text)))
    length = min(len(text), 2600) / 2600
    format_ok = 1.0 if text.startswith('[') and markers else 0.0
    return (1.5 * cover + 1.0 * format_ok + 0.6 * length + 0.05 * min(actions, 12)
            - 3.0 * repetition - 6.0 * junk_ratio(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=0.8)
    args = parser.parse_args()

    import torch
    from music_planner_adapters import load_planner_adapter

    manifest = json.loads((args.benchmark / 'manifest.json').read_text())[:args.limit]
    adapter = load_planner_adapter(family='omni_captioner', model_path=args.model,
                                   checkpoint=args.checkpoint, device_map='auto', dtype=torch.bfloat16)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    results = []
    with (args.out / 'samples.jsonl').open('w', encoding='utf-8') as stream:
        for case in manifest:
            audio = (args.benchmark / case['audio']).resolve()
            row = dict(id=case['id'], source_group_id=case['id'], audio_path=str(audio),
                       audio_sha256=sha256_file(audio), audio_window_seconds=[0.0, 5.0625])
            candidates = []
            for index in range(args.samples):
                batch = adapter.prefix(row)
                adapter.model.eval()
                parameter = next(adapter.model.parameters())
                with torch.no_grad(), torch.autocast(parameter.device.type, dtype=torch.bfloat16,
                                                     enabled=parameter.dtype == torch.bfloat16):
                    output = adapter.model.generate(
                        **adapter._move(batch), do_sample=True, temperature=args.temperature, top_p=0.95,
                        num_beams=1, repetition_penalty=1.05, no_repeat_ngram_size=4,
                        max_new_tokens=1024, use_cache=True)
                sequences = output.sequences if hasattr(output, 'sequences') else output
                ids = sequences[0, batch['input_ids'].shape[1]:].cpu().tolist()
                raw = adapter.tokenizer.decode(ids, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
                text = clean(raw)
                entry = dict(index=index, raw=raw, text=text, score=score(text))
                candidates.append(entry)
                stream.write(json.dumps(dict(case_id=case['id'], **entry), ensure_ascii=False) + '\n')
                stream.flush()
            best = max(candidates, key=lambda item: item['score'])
            results.append(dict(id=case['id'], genre=case['genre'], best=best,
                                scores=[round(c['score'], 4) for c in candidates]))
            print(json.dumps(dict(id=case['id'], scores=[round(c['score'], 3) for c in candidates],
                                  best=round(best['score'], 3))), flush=True)
    summary = dict(cases=len(results), samples=args.samples,
                   mean_score=sum(r['best']['score'] for r in results) / len(results),
                   rows=results)
    (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(cases=len(results), mean_score=summary['mean_score']), indent=2))


if __name__ == '__main__':
    main()
