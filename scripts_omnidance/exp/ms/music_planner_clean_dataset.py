"""Create a new shared benchmark by excluding the seven audited invalid train sources."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(value):
    return hashlib.sha256(value).hexdigest()


def reference(path, value):
    return dict(path=str(path), sha256=digest(value), size_bytes=len(value))


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write(chr(10))


def clean_dataset(parent, audit_path, out):
    if out.exists():
        raise FileExistsError('Preserve existing output: ' + str(out))
    summary_bytes = (parent / 'summary.json').read_bytes()
    summary = json.loads(summary_bytes)
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    if (summary['split_counts'] != dict(train=10720, val=490, test=500) or
            audit['complete'] is not True or audit['summary']['train']['audited'] != 10720 or
            audit['summary']['val']['audited'] != 490 or
            audit['summary']['val']['old_length_gate_rejected']):
        raise ValueError('Parent baseline/header audit does not match the approved cleanup')
    invalid = audit['summary']['train']['old_length_gate_rejected']
    excluded = {row['id'] for row in invalid}
    if len(invalid) != 7 or len(excluded) != 7:
        raise ValueError('Only the seven audited train exclusions are approved')
    data = {split: (parent / (split + '.jsonl')).read_bytes() for split in ('train', 'val', 'test')}
    for split, value in data.items():
        if digest(value) != summary['split_artifacts'][split]['sha256']:
            raise ValueError('Parent split bytes changed: ' + split)
    for split in ('train', 'val'):
        if digest(data[split]) != audit['split_refs'][split]['sha256']:
            raise ValueError('Audited split differs: ' + split)
    kept, removed = [], []
    by_id = {row['id']: row for row in invalid}
    for number, line in enumerate(data['train'].splitlines(keepends=True), 1):
        if not line.strip():
            kept.append(line)
            continue
        row = json.loads(line)
        if row['id'] not in excluded:
            kept.append(line)
            continue
        evidence = by_id[row['id']]
        if row['split'] != 'train' or evidence['line'] != number or evidence['audio_path'] != row['audio_path']:
            raise ValueError('Audited exclusion membership/order differs')
        removed.append(dict(id=row['id'], parent_line=number, audio_sha256=row['audio_sha256'],
            frames=evidence['frames'], sample_rate=evidence['sample_rate'], extra_frames=evidence['extra_frames'],
            reason='original strict complete-audio length gate rejection'))
    if len(removed) != 7:
        raise ValueError('Missing or repeated exclusion')
    data['train'] = b''.join(kept)
    counts = dict(train=10713, val=490, test=500)
    if any(sum(bool(line.strip()) for line in value.splitlines()) != counts[split]
           for split, value in data.items()):
        raise ValueError('Clean split counts differ')
    out.mkdir()
    for split, value in data.items():
        with (out / (split + '.jsonl')).open('xb') as stream:
            stream.write(value)
    exclusion = dict(schema='music_planner_training_exclusion_v1',
        authorization='User approved excluding seven invalid train sources',
        parent_summary_ref=reference(parent / 'summary.json', summary_bytes),
        header_audit_ref=reference(audit_path, audit_bytes), removed=removed, exclusion_count=7,
        original_train_count=10720, clean_train_count=10713, removed_train_only=True,
        retained_train_line_bytes_and_order_unchanged=True, val_and_test_bytes_unchanged=True,
        original_audio_modified=False, original_dataset_modified=False,
        whole_waveform_content_SHA_reverified=False, test_generation_or_selection_performed=False,
        GPU_training_started=False)
    write_json(out / 'exclusion_receipt.json', exclusion)
    exclusion_ref = reference(out / 'exclusion_receipt.json', (out / 'exclusion_receipt.json').read_bytes())
    cleaned = dict(summary)
    cleaned.update(parent_complete_sources=summary['complete_sources'], complete_sources=sum(counts.values()),
        excluded_invalid_audio_train_sources=7, split_counts=counts,
        split_artifacts={split: dict(path=str(out / (split + '.jsonl')), sha256=digest(value))
                         for split, value in data.items()}, output=str(out),
        parent_baseline_ref=exclusion['parent_summary_ref'], training_exclusion_receipt_ref=exclusion_ref,
        training_started=False, semantic_approval=False, human_review_completed=False)
    write_json(out / 'summary.json', cleaned)
    write_json(out / 'training_contract.json', dict(schema='music_planner_clean_training_contract_v1',
        families=['music_flamingo', 'omni_captioner', 'moss'], same_dataset_for_all_families=True,
        split_counts=counts, epochs=2, world_size=8, microbatch=1, gradient_accumulation=4, global_batch=32,
        updates_per_epoch=math.ceil(10713 / 32), total_optimizer_updates=2 * math.ceil(10713 / 32),
        source_exposure_budget=2 * 10713, last_batch_sources_per_epoch=10713 % 32,
        full_audio_window_seconds=[0, 5.0625], caption_windows_seconds=[[0, 2], [2, 5.0625]],
        preserve_original_strict_audio_gate=True, fresh_official_base_required=True,
        resume_old_fingerprint_allowed=False, initial_updates_2_and_3_counted_in_formal_budget=True,
        split_artifacts=cleaned['split_artifacts'], exclusion_receipt_ref=exclusion_ref, GPU_training_started=False))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    counts = clean_dataset(args.parent, args.audit, args.out)
    print(json.dumps(dict(out=str(args.out), split_counts=counts, original_media_modified=False,
                          GPU_training_started=False)))


if __name__ == '__main__':
    main()
