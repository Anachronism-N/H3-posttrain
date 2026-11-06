#!/usr/bin/env python3
"""Build full-v4 SFT data from the final 220,667-caption delivery.

Validates exact 24 kHz mono 121,500-frame audio, strips optional style headers,
and honors the delivered split when present (train/valtest), otherwise derives a
source-group-isolated 90/5/5 split.
"""
import argparse
import hashlib
import json
import os
import wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--merged', type=Path, nargs='+', required=True)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('exists: ' + str(args.out))

    rows, rejected = [], Counter()
    lines = []
    for path in args.merged:
        lines.extend(path.read_text(encoding='utf-8').splitlines())
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        sid, text = record.get('id'), record.get('motion_caption')
        if not isinstance(sid, str) or not isinstance(text, str) or not text.strip():
            rejected['schema'] += 1
            continue
        if len(text) > 10000:
            rejected['over_10k'] += 1
            continue
        if '<|' in text or '<audio>' in text:
            rejected['reserved'] += 1
            continue
        if text.startswith('Dance style:'):
            parts = text.split('\n', 1)
            if len(parts) != 2 or not parts[1].strip():
                rejected['empty_body'] += 1
                continue
            text = parts[1]
        rows.append(dict(id=sid, text=text, group=record.get('source_group_id'),
                         source_split=record.get('split')))

    def check(row):
        path = args.audio / (row['id'] + '.wav')
        try:
            with wave.open(str(path), 'rb') as handle:
                if (handle.getframerate(), handle.getnchannels(), handle.getnframes()) != (24000, 1, 121500):
                    return None, 'geometry'
            return dict(row, sha=sha_file(path)), None
        except Exception:
            return None, 'unreadable'

    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(check, rows))
    kept = []
    for row, error in results:
        if error:
            rejected[error] += 1
        else:
            kept.append(row)

    parent = {}

    def find(value):
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for row in kept:
        union(('id', row['id']), ('g', row['group'] or ''))
        union(('id', row['id']), ('a', row['sha']))
    cluster_split, conflicts = {}, 0
    for row in kept:
        delivered = row.get('source_split')
        if delivered == 'valtest':
            delivered = 'test'
        if delivered not in ('train', 'val', 'test'):
            continue
        cluster = find(('id', row['id']))
        if cluster in cluster_split and cluster_split[cluster] != delivered:
            conflicts += 1
        cluster_split.setdefault(cluster, delivered)
    if conflicts:
        raise SystemExit('delivered split conflicts across source/audio clusters: %d' % conflicts)
    clusters = sorted({find(('id', row['id'])) for row in kept},
                      key=lambda value: hashlib.sha256((str(args.seed) + ':' + str(value)).encode()).hexdigest())
    n = len(clusters)
    nt, nv = int(n * 0.9), int(n * 0.05)
    for index, cluster in enumerate(clusters):
        if cluster not in cluster_split:
            cluster_split[cluster] = 'train' if index < nt else 'val' if index < nt + nv else 'test'
    for row in kept:
        row['split'] = cluster_split[find(('id', row['id']))]
        if row['split'] == 'valtest':
            row['split'] = 'test'

    args.out.mkdir(parents=True)
    (args.out / 'audio').mkdir()
    for row in kept:
        os.link(args.audio / (row['id'] + '.wav'), args.out / 'audio' / (row['id'] + '.wav'))
    for split in ('train', 'val', 'test'):
        values = [r for r in kept if r['split'] == split]
        data = [dict(id=r['id'], source_group_id=r['group'], split=split,
                     audio_path='audio/' + r['id'] + '.wav', audio_sha256=r['sha'],
                     audio_duration_seconds=5.0625, audio_window_seconds=[0.0, 5.0625],
                     target_text=r['text'], caption_version='m2d-full-v4',
                     semantic_approval=False, human_review_completed=False) for r in values]
        (args.out / (split + '.jsonl')).write_text(
            ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in data), encoding='utf-8')
        print(split, len(data))
    contract = dict(schema='music-planner-caption-contract-v1', caption_version='m2d-full-v4',
                    task_instruction='Listen to the music and describe the dance.',
                    audio_min_seconds=1.0, audio_max_seconds=10.0,
                    required_markers=[], require_target_windows=False)
    (args.out / 'caption_contract.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2) + '\n')
    (args.out / 'caption_contract_nostyle.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2) + '\n')
    summary = dict(schema='m2d-full-v4', input=len(rows) + rejected['schema'],
                   kept=len(kept), rejected=dict(rejected),
                   split_counts={s: sum(1 for r in kept if r['split'] == s) for s in ('train', 'val', 'test')},
                   audio_geometry='24kHz mono 121500', audio_sha_verified=True, nostyle=True)
    (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
