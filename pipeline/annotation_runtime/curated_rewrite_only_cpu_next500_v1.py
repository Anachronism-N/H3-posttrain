"""Finite physical-row continuation of the frozen zero-model facts replay."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

RT = Path(__file__).resolve().parent
BASE = RT / 'curated_rewrite_only_cpu_batch_v1.py'
BASE_SHA = '0f8c684daa401c8a9a7deac3f1b6f86c166c45a8cd9a610a5ca8b1b10fe5290f'
OUT = RT.parent / 'curated_rewrite_only_cpu_next500_candidates_v1'
START = 500
LIMIT = 500


def base_module():
    raw = BASE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_SHA:
        raise ValueError('Frozen zero-model source replay changed')
    spec = importlib.util.spec_from_file_location('next500_frozen_cpu_replay', BASE)
    value = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(BASE), 'exec'), value.__dict__)
    return value


def read_window(path, start=START, limit=LIMIT):
    if start != START or limit != LIMIT:
        raise ValueError('Only the next fixed physical rows500..999 are allowed')
    path = Path(path)
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        lines = []
        for _ in range(start + limit):
            line = stream.readline()
            if not line or not line.endswith(b'\n'):
                break
            lines.append(line)
        if len(lines) < start + limit:
            raise ValueError('Full fixed next500 window not ready; no partial selection published')
        prefix = b''.join(lines)
        stream.seek(0)
        after = os.fstat(stream.fileno())
        current = path.stat()
        if ((before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
                or (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino)
                or current.st_size < len(prefix) or stream.read(len(prefix)) != prefix):
            raise ValueError('Original ready prefix changed/replaced/truncated')
    selected = lines[start:start + limit]
    offset = sum(map(len, lines[:start]))
    return [json.loads(line) for line in selected], {
        'path': str(path.resolve()), 'prefix_bytes': len(prefix), 'prefix_lines': len(lines),
        'prefix_sha256': hashlib.sha256(prefix).hexdigest(), 'physical_start_row_zero_based': start,
        'selected_physical_rows': limit, 'selected_byte_offset': offset,
        'selected_byte_length': sum(map(len, selected)),
        'selected_raw_sha256': hashlib.sha256(b''.join(selected)).hexdigest(),
        'earlier_original_prefix_preserved': True, 'native_first500_not_reprocessed': True}


def run():
    base = base_module()
    original = base.stable_prefix
    base.stable_prefix = lambda limit: read_window(base.QUEUE / 'ready.jsonl', START, limit)
    try:
        report = base.once(LIMIT, out=OUT)
    finally:
        base.stable_prefix = original
    receipt = {'schema': 'curated_zero_model_next500_execution_receipt_v1',
        'original_replay_ref': {'path': str(BASE), 'sha256': BASE_SHA},
        'execution_ref': base.q.identity(__file__), 'original_ready_prefix': report['original_ready_prefix'],
        'source_dedup_count': report['source_dedup_count'], 'joint_complete_count': report['joint_complete_count'],
        'pending_reason_counts': report['pending_reason_counts'],
        'candidate_manifest_ref': base.q.identity(OUT / 'revisions' / base.REVISION / 'review-required.manifest.json'),
        'model_calls': 0, 'attempt_budget_consumed': False, 'original_population_quality_approved': False,
        'independent_review_required': True, **base.q.FLAGS}
    base.q.save_once(OUT / 'execution.receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(run(), ensure_ascii=False), flush=True)
