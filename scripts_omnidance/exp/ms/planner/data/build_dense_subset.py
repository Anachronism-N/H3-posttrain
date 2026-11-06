#!/usr/bin/env python3
"""Select the densest full-v4 training captions for a density-focused SFT run."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

MARKER = re.compile(r'\[\d+(?:\.\d+)?\s*[-–—]\s*\d+(?:\.\d+)?\s*s?\]')


def density(text):
    parts = [part for part in MARKER.split(text) if part.strip()]
    clauses = [s for part in parts for s in re.split(r'[;。]', part) if s.strip()]
    return len(clauses) / max(len(parts), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train', type=int, default=10000)
    parser.add_argument('--val', type=int, default=500)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('exists')
    args.out.mkdir(parents=True)
    (args.out / 'audio').mkdir()

    def load(split):
        rows = [json.loads(line) for line in (args.data / (split + '.jsonl')).read_text().splitlines() if line.strip()]
        rows.sort(key=lambda row: (-len(row['target_text']), -density(row['target_text']),
                                   hashlib.sha256(row['id'].encode()).hexdigest()))
        return rows

    train = load('train')[:args.train]
    val = load('val')[:args.val]
    for split, rows in (('train', train), ('val', val)):
        for row in rows:
            os.link(args.data / row['audio_path'], args.out / 'audio' / (row['id'] + '.wav'))
        for row in rows:
            row['split'] = split
        (args.out / (split + '.jsonl')).write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
        print(split, len(rows), 'mean density', sum(density(r['target_text']) for r in rows) / len(rows),
              'mean chars', sum(len(r['target_text']) for r in rows) / len(rows))
    contract = json.loads((args.data / 'caption_contract_nostyle.json').read_text())
    contract['caption_version'] = 'm2d-full-v4-dense-v1'
    (args.out / 'caption_contract.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
