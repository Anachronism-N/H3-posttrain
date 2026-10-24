"""Read-only, bounded-concurrency train/val WAV header audit; no GPU quality approval."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import wave


def inspect_audio(item):
    split, number, row = item
    path = os.path.abspath(os.path.expanduser(row['audio_path']))
    result = dict(split=split, line=number, id=row['id'], audio_path=path)
    try:
        with wave.open(path, 'rb') as stream:
            frames, rate = stream.getnframes(), stream.getframerate()
            channels, sample_width = stream.getnchannels(), stream.getsampwidth()
        extra = frames - 5.0625 * rate
        old_rejects = abs(frames / rate - 5.0625) > 1 / rate
        candidate_accepts = not old_rejects or (rate == 24000 and 0 < extra <= 5)
        result.update(frames=frames, sample_rate=rate, channels=channels, sample_width=sample_width,
                      extra_frames=extra, duration_seconds=frames / rate,
                      old_length_gate_accepts=not old_rejects,
                      candidate_length_gate_accepts=candidate_accepts,
                      declared_full_window=row['audio_window_seconds'] == [0., 5.0625])
    except (OSError, EOFError, wave.Error, ValueError) as error:
        result['error'] = dict(type=type(error).__name__, message=str(error))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Audit output must be new; never replace earlier evidence')
    items, split_refs = [], {}
    for split, expected in (('train', 10720), ('val', 490)):
        path = args.data / (split + '.jsonl')
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if len(rows) != expected or any(row['split'] != split for row in rows):
            raise ValueError('Fixed baseline split count/membership differs')
        split_refs[split] = dict(path=str(path), count=len(rows), sha256=hashlib.sha256(raw).hexdigest())
        items.extend((split, index, row) for index, row in enumerate(rows, 1))
    counts = {split: Counter() for split in split_refs}
    rows_by_split = {split: [] for split in split_refs}
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(inspect_audio, item) for item in items]
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows_by_split[row['split']].append(row)
            if 'error' not in row:
                counts[row['split']][(row['sample_rate'], row['channels'], row['sample_width'], row['extra_frames'])] += 1
            if completed % 500 == 0 or completed == len(items):
                print(json.dumps(dict(audited=completed, total=len(items), elapsed_seconds=time.monotonic() - started)), flush=True)
    summary = {}
    failures = []
    for split, rows in rows_by_split.items():
        rows.sort(key=lambda row: row['line'])
        old_rejected = [row for row in rows if row.get('old_length_gate_accepts') is False]
        rejected = [row for row in rows if row.get('error') or row.get('candidate_length_gate_accepts') is not True
                    or row.get('declared_full_window') is not True or row.get('channels') != 1]
        failures.extend(rejected)
        summary[split] = dict(audited=len(rows), distribution=[dict(sample_rate=key[0], channels=key[1],
            sample_width=key[2], extra_frames=key[3], count=count) for key, count in sorted(counts[split].items())],
            old_length_gate_rejected=old_rejected, candidate_rejected=rejected)
    receipt = dict(schema='music-planner-audio-header-audit-v1', workers=4, split_refs=split_refs,
        summary=summary, complete=True, candidate_header_gate_passed=not failures,
        waveform_content_SHA_reverified=False, waveform_finite_values_checked=False,
        test_used=False, original_waveform_modified=False, original_split_modified=False,
        GPU_training_executed=False, semantic_quality_accepted=False,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        elapsed_seconds=time.monotonic() - started)
    with args.out.open('x') as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(out=str(args.out), complete=True, candidate_header_gate_passed=not failures,
                         failures=len(failures))), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
