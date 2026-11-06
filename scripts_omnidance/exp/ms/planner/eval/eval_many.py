#!/usr/bin/env python3
"""Generate captions for N official test rows and compare with references."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

MS = Path('/ssd/2/niezichao.nzc/artifacts/h3-jev-worktree-120fe6d/scripts_omnidance/exp/ms')
sys.path.insert(0, str(MS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fidelity import coverage, tags, ACTION, BODY  # noqa: E402

STOPS = ('\nassistant', '\nAssistant', 'Assistant:', 'End of interaction', '###', '<|im_end|>')
JUNK_LINE = re.compile(r'^[-|=#.:!?@%$*+\[\]{}<>~^&(),;\s]+$')


def clean(text):
    cut = len(text)
    for token in STOPS:
        index = text.find(token)
        if index != -1:
            cut = min(cut, index)
    kept = []
    for line in text[:cut].splitlines():
        stripped = line.strip()
        if not stripped or re.search(r'[\u4e00-\u9fff]', stripped) or JUNK_LINE.fullmatch(stripped):
            break
        if len(stripped) < 12 and not stripped.startswith('['):
            break
        kept.append(stripped)
    return '\n'.join(kept).strip()


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
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--rows', type=int, default=30)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device-map', default='auto')
    parser.add_argument('--min-new-tokens', type=int, default=0)
    parser.add_argument('--eos-until-coverage', action='store_true')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    args = parser.parse_args()

    import torch
    from music_planner_adapters import load_planner_adapter

    adapter = load_planner_adapter(family='omni_captioner', model_path=args.model,
                                   checkpoint=args.checkpoint, device_map=args.device_map,
                                   dtype=torch.bfloat16)
    rows = []
    with (args.data / (args.split + '.jsonl')).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda row: hashlib.sha256(('many:' + row['id']).encode()).hexdigest())
    rows = rows[:args.rows][args.shard_index::args.shard_count]
    args.out.mkdir(parents=True, exist_ok=False)
    metrics = []
    with (args.out / 'generations.jsonl').open('w', encoding='utf-8') as stream:
        for index, row in enumerate(rows):
            audio = Path(row['audio_path'])
            if not audio.is_absolute():
                audio = (args.data / audio).resolve()
            source = dict(id=row['id'], source_group_id=row['source_group_id'], audio_path=str(audio),
                          audio_sha256=row['audio_sha256'], audio_window_seconds=[0.0, 5.0625])
            batch = adapter.prefix(source)
            adapter.model.eval()
            parameter = next(adapter.model.parameters())
            processors = None
            if args.eos_until_coverage:
                from transformers import LogitsProcessor

                class CoverageEos(LogitsProcessor):
                    def __init__(self, tokenizer, eos_ids):
                        self.tokenizer = tokenizer
                        self.eos_ids = list(eos_ids)
                        self.covered = False

                    def __call__(self, input_ids, scores):
                        if not self.covered:
                            text = self.tokenizer.decode(input_ids[0, -300:], skip_special_tokens=True)
                            ends = [float(end) for _, end in re.findall(
                                r'\[(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)\s*s?\]', text)]
                            if ends and max(ends) >= 5.0:
                                self.covered = True
                            else:
                                scores[:, self.eos_ids] = float('-inf')
                        return scores

                processors = [CoverageEos(adapter.tokenizer, adapter.eos_token_ids())]
            with torch.no_grad(), torch.autocast(parameter.device.type, dtype=torch.bfloat16,
                                                 enabled=parameter.dtype == torch.bfloat16):
                output = adapter.model.generate(**adapter._move(batch), do_sample=False, num_beams=1,
                                                repetition_penalty=1.05, no_repeat_ngram_size=4,
                                                logits_processor=processors,
                                                min_new_tokens=(args.min_new_tokens or None),
                                                max_new_tokens=1024, use_cache=True)
            sequences = output.sequences if hasattr(output, 'sequences') else output
            ids = sequences[0, batch['input_ids'].shape[1]:].cpu().tolist()
            text = clean(adapter.tokenizer.decode(ids, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False))
            reference = row['target_text']
            record = dict(id=row['id'], text=text, reference=reference,
                          coverage=coverage(text), reference_coverage=coverage(reference),
                          action_jaccard=len(tags(text, ACTION) & tags(reference, ACTION)) /
                                         max(len(tags(text, ACTION) | tags(reference, ACTION)), 1),
                          body_jaccard=len(tags(text, BODY) & tags(reference, BODY)) /
                                       max(len(tags(text, BODY) | tags(reference, BODY)), 1),
                          length_ratio=len(text) / max(len(reference), 1))
            metrics.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            if (index + 1) % 5 == 0:
                print('done %d/%d' % (index + 1, len(rows)), flush=True)
    summary = dict(rows=len(metrics),
                   mean_coverage=sum(r['coverage'] for r in metrics) / len(metrics),
                   mean_reference_coverage=sum(r['reference_coverage'] for r in metrics) / len(metrics),
                   mean_action_jaccard=sum(r['action_jaccard'] for r in metrics) / len(metrics),
                   mean_body_jaccard=sum(r['body_jaccard'] for r in metrics) / len(metrics),
                   mean_length_ratio=sum(r['length_ratio'] for r in metrics) / len(metrics))
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
