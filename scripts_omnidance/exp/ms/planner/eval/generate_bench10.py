#!/usr/bin/env python3
"""Generate planner captions for the fixed benchmark10 cases.

Loads one trained adapter checkpoint (adapter + optional JEV head), runs greedy
native audio-only generation per case, and writes raw outputs plus light
format/repetition metrics. No reference caption is passed to the model.
"""
import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

MS = Path('/ssd/2/niezichao.nzc/artifacts/h3-jev-worktree-120fe6d/scripts_omnidance/exp/ms')
sys.path.insert(0, str(MS))

MARKER = re.compile(r'\[(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)\s*s?\]')
STYLE = re.compile(r'^\s*Dance style:\s*(.+?)\s*$', re.I | re.M)
ACTION_PATTERNS = {
    'step': r'\b(?:steps?|stepp?ing|stepped|walk\w*|footwork)\b|踏步|迈步',
    'turn': r'\b(?:turn\w*|pivot\w*|spin\w*|rotat\w*)\b|转身|旋转',
    'sway': r'\b(?:sway\w*|rock\w*|shift\w*)\b|摇摆|重心',
    'bend': r'\b(?:bend\w*|bent|crouch\w*|squat\w*|dip\w*)\b|屈膝|下蹲',
    'extend': r'\b(?:extend\w*|stretch\w*|reach\w*|open\w*)\b|伸展|展开',
    'lift': r'\b(?:lift\w*|rais\w*|jump\w*|hop\w*)\b|抬起|跳跃',
    'lower': r'\b(?:lower\w*|drop\w*)\b|放下|降低',
    'cross': r'\b(?:cross\w*|uncross\w*)\b|交叉',
    'sweep': r'\b(?:sweep\w*|swept|swing\w*|swung|wave\w*)\b|挥动|摆臂',
}
BODY_PATTERNS = {
    'arms': r'\b(?:arms?|elbows?|shoulders?)\b|手臂|肩|肘',
    'hands': r'\b(?:hands?|palms?|wrists?|fingers?)\b|手掌|手腕|手指',
    'legs': r'\b(?:legs?|knees?|thighs?)\b|腿|膝',
    'feet': r'\b(?:feet|foot|toes?|heels?)\b|脚|足',
    'torso': r'\b(?:torso|hips?|waist|chest|spine|trunk)\b|躯干|腰|胯|胸',
    'head': r'\b(?:head|neck|chin)\b|头|颈',
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ngram_repetition(text, n=8):
    tokens = re.findall(r"[a-z]+(?:['’][a-z]+)?|[\u3400-\u9fff]|\d+(?:\.\d+)?", text.lower())
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    return 1.0 - len(counts) / len(grams)


def metrics(text):
    stripped = STYLE.sub('', text).strip()
    markers = MARKER.findall(stripped)
    style = STYLE.search(text)
    actions = [name for name, pattern in ACTION_PATTERNS.items() if re.search(pattern, stripped, re.I)]
    body = [name for name, pattern in BODY_PATTERNS.items() if re.search(pattern, stripped, re.I)]
    sentences = [part for part in re.split(r'[;。\n]', stripped) if part.strip()]
    return dict(
        char_length=len(text),
        style_header=style.group(1) if style else None,
        window_count=len(markers),
        action_types=actions,
        body_parts=body,
        unique_sentence_ratio=(len(set(sentences)) / len(sentences)) if sentences else 0.0,
        ngram8_repetition=round(ngram_repetition(stripped), 4),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, help='Omit with --base-only for the zero-shot captioner')
    parser.add_argument('--base-only', action='store_true')
    parser.add_argument('--jev', type=Path, help='Exported JEV head directory')
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--repetition-penalty', type=float, default=1.0)
    parser.add_argument('--no-repeat-ngram', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='0 keeps greedy decoding; >0 samples with this temperature')
    parser.add_argument('--val-data', type=Path, help='Data dir with val.jsonl for a teacher-forced NLL check')
    parser.add_argument('--val-limit', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--device-map', help="accelerate device_map, e.g. 'auto' to shard across visible GPUs")
    args = parser.parse_args()

    import torch
    from music_planner_adapters import load_planner_adapter

    manifest = json.loads((args.benchmark / 'manifest.json').read_text())
    if args.limit:
        manifest = manifest[:args.limit]
    args.out.mkdir(parents=True, exist_ok=False)
    adapter = load_planner_adapter(family='omni_captioner', model_path=args.model,
                                   checkpoint=None if args.base_only else args.checkpoint,
                                   device=args.device if not args.device_map else None,
                                   device_map=args.device_map, dtype=torch.bfloat16)
    if args.jev is not None:
        from music_planner_jev import load_jev
        load_jev(adapter, args.jev)

    validation = None
    if args.val_data is not None:
        import torch as _torch
        losses = []
        with (args.val_data / 'val.jsonl').open(encoding='utf-8') as stream:
            for index, line in enumerate(stream):
                if index >= args.val_limit:
                    break
                row = json.loads(line)
                audio = Path(row['audio_path'])
                if not audio.is_absolute():
                    audio = (args.val_data / audio).resolve()
                row = dict(row, audio_path=str(audio))
                result = adapter.teacher_forced(row)
                losses.append(result['mean_nll'])
                print('val NLL', row['id'], round(result['mean_nll'], 4), flush=True)
        validation = dict(mean_nll=sum(losses) / len(losses), rows=len(losses))

    rows = []
    for case in manifest:
        audio = (args.benchmark / case['audio']).resolve()
        row = {'id': case['id'], 'source_group_id': case['id'], 'audio_path': str(audio),
               'audio_sha256': sha256_file(audio), 'audio_window_seconds': [0.0, 5.0625]}
        started = time.monotonic()
        record = dict(id=case['id'], genre=case['genre'], difficulty=case['difficulty'],
                      reference_prompt=case.get('prompt'), audio=str(audio),
                      decoding=dict(repetition_penalty=args.repetition_penalty,
                                    temperature=args.temperature))
        try:
            if args.repetition_penalty == 1.0 and args.temperature == 0.0 and args.no_repeat_ngram == 0:
                result = adapter.generate(row, max_new_tokens=args.max_new_tokens)
            else:
                import torch as _torch
                batch = adapter.prefix(row)
                adapter.model.eval()
                parameter = next(adapter.model.parameters())
                with _torch.no_grad(), _torch.autocast(
                        parameter.device.type, dtype=_torch.bfloat16,
                        enabled=parameter.dtype == _torch.bfloat16):
                    output = adapter.model.generate(
                        **adapter._move(batch), do_sample=args.temperature > 0,
                        temperature=args.temperature if args.temperature > 0 else None,
                        num_beams=1, repetition_penalty=args.repetition_penalty,
                        no_repeat_ngram_size=args.no_repeat_ngram, max_new_tokens=args.max_new_tokens,
                        use_cache=True)
                sequences = output.sequences if hasattr(output, 'sequences') else output
                new_ids = sequences[0, batch['input_ids'].shape[1]:].cpu().tolist()
                eos = adapter.eos_token_ids()
                result = dict(raw_text=adapter.tokenizer.decode(
                    new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                    new_token_ids=new_ids,
                    stop_reason='eos' if new_ids and new_ids[-1] in eos else 'max_new_tokens')
            record['raw_text'] = result['raw_text']
            record['generated_tokens'] = len(result['new_token_ids'])
            record['stop_reason'] = result['stop_reason']
            record['metrics'] = metrics(result['raw_text'])
            if args.jev is not None:
                from music_planner_jev import predict_jev
                head = predict_jev(adapter, row)
                probs = head['probabilities'][0].tolist()
                ranked = sorted(zip(head['classes'], probs), key=lambda item: -item[1])
                record['jev'] = dict(classes=head['classes'], probabilities=probs,
                                     top1=ranked[0][0], top1_prob=ranked[0][1], top3=ranked[:3])
        except Exception as error:
            record['failure'] = dict(type=type(error).__name__, message=str(error))
        record['elapsed_seconds'] = round(time.monotonic() - started, 2)
        rows.append(record)
        with (args.out / 'generations.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
        print(json.dumps({k: record.get(k) for k in ('id', 'genre', 'difficulty', 'generated_tokens',
                                                     'stop_reason', 'failure')}, ensure_ascii=False), flush=True)

    summary = dict(checkpoint=str(args.checkpoint), jev=str(args.jev) if args.jev else None,
                   validation_nll=validation,
                   cases=len(rows), failures=sum('failure' in row for row in rows),
                   mean_chars=sum(row.get('metrics', {}).get('char_length', 0) for row in rows) / max(len(rows), 1),
                   mean_windows=sum(row.get('metrics', {}).get('window_count', 0) for row in rows) / max(len(rows), 1),
                   mean_ngram8_repetition=sum(row.get('metrics', {}).get('ngram8_repetition', 0) for row in rows) / max(len(rows), 1),
                   styled_cases=sum(bool(row.get('metrics', {}).get('style_header')) for row in rows),
                   jev_top1_match=sum(1 for row in rows if row.get('jev') and row['jev']['top1'] == row['genre']),
                   jev_scored=sum(1 for row in rows if row.get('jev')))
    (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
