"""Bounded continuation windows of the verified zero-model source replay."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

RT = Path(__file__).resolve().parent
WINDOW = RT / 'curated_rewrite_only_cpu_next500_v1.py'
WINDOW_SHA = '06bbfebadfe41680cc4a92161299cda860950bbce5f147e72a4f4ef1115caf92'


def validate_start(start):
    if type(start) is not int or start < 1000:
        raise ValueError('Continuation must start after the first1000 physical rows')


def run(start, count):
    validate_start(start)
    if type(count) is not int or not 1 <= count <= 500:
        raise ValueError('Continuation count must be a fixed bounded1..500 window')
    raw = WINDOW.read_bytes()
    if hashlib.sha256(raw).hexdigest() != WINDOW_SHA:
        raise ValueError('Verified original bounded-window reader changed')
    spec = importlib.util.spec_from_file_location('continuation_private_window', WINDOW)
    window = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(WINDOW), 'exec'), window.__dict__)
    out = RT.parent / ('curated_rewrite_only_cpu_continuation_%06d_count%03d_v1' % (start, count))
    if (out / 'execution.receipt.json').exists():
        raise ValueError('This continuation window is already published; no duplicate run')
    window.START, window.LIMIT, window.OUT = start, count, out
    # Original read_window still checks the fixed selected count and its complete
    # preceding physical prefix; no original ready bytes or source budget change.
    report = window.run()
    base = window.base_module()
    receipt = {**report, 'schema': 'curated_zero_model_continuation_execution_v1',
               'continuation_execution_ref': base.q.identity(__file__),
               'verified_window_reader_ref': {'path': str(WINDOW), 'sha256': WINDOW_SHA},
               'physical_start_row_zero_based': start, 'maximum_selected_physical_rows': 500,
               'selected_physical_rows': count, 'next_unprocessed_physical_row': start + count,
               'original500_batch_membership_unchanged': True,
               'earlier_windows_not_reprocessed': True}
    base.q.save_once(out / 'continuation.receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--count', type=int, default=500)
    parser.add_argument('--once', action='store_true', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.start, args.count), ensure_ascii=False), flush=True)
