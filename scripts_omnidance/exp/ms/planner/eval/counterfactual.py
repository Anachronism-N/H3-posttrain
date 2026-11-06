#!/usr/bin/env python3
"""Swap audio between holdout cases to test whether captions depend on the music."""
import argparse
import difflib
import hashlib
import json
import sys
from pathlib import Path

MS = Path('/ssd/2/niezichao.nzc/artifacts/h3-jev-worktree-120fe6d/scripts_omnidance/exp/ms')
sys.path.insert(0, str(MS))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--originals', type=Path, required=True,
                        help='generations_clean.jsonl from the same checkpoint')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=5)
    args = parser.parse_args()

    import torch
    from music_planner_adapters import load_planner_adapter

    manifest = json.loads((args.benchmark / 'manifest.json').read_text())[:args.limit]
    originals = {}
    for line in args.originals.open(encoding='utf-8'):
        row = json.loads(line)
        originals[row['id']] = row['raw_text']
    adapter = load_planner_adapter(family='omni_captioner', model_path=args.model,
                                   checkpoint=args.checkpoint, device_map='auto', dtype=torch.bfloat16)
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, case in enumerate(manifest):
        donor = manifest[(index + 1) % len(manifest)]
        audio = (args.benchmark / donor['audio']).resolve()
        row = dict(id=case['id'], source_group_id=case['id'], audio_path=str(audio),
                   audio_sha256=sha256_file(audio), audio_window_seconds=[0.0, 5.0625])
        batch = adapter.prefix(row)
        adapter.model.eval()
        parameter = next(adapter.model.parameters())
        with torch.no_grad(), torch.autocast(parameter.device.type, dtype=torch.bfloat16,
                                             enabled=parameter.dtype == torch.bfloat16):
            output = adapter.model.generate(**adapter._move(batch), do_sample=False, num_beams=1,
                                            repetition_penalty=1.05, no_repeat_ngram_size=4,
                                            max_new_tokens=1024, use_cache=True)
        sequences = output.sequences if hasattr(output, 'sequences') else output
        ids = sequences[0, batch['input_ids'].shape[1]:].cpu().tolist()
        text = adapter.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        original = originals[case['id']]
        rows.append(dict(id=case['id'], genre=case['genre'], donor_id=donor['id'],
                         donor_genre=donor['genre'], original=original, swapped=text,
                         similarity_to_original=difflib.SequenceMatcher(
                             None, original[:1200].lower(), text[:1200].lower()).ratio()))
        with (args.out / 'pairs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(rows[-1], ensure_ascii=False) + '\n')
        print(json.dumps({k: rows[-1][k] for k in ('id', 'donor_id', 'similarity_to_original')}), flush=True)
    summary = dict(cases=len(rows),
                   mean_similarity=sum(r['similarity_to_original'] for r in rows) / len(rows),
                   min_similarity=min(r['similarity_to_original'] for r in rows),
                   max_similarity=max(r['similarity_to_original'] for r in rows))
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
