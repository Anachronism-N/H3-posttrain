"""Independent full-data entry reusing the frozen pilot's training implementation."""
import argparse
from datetime import timedelta
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import math
from pathlib import Path
import sys
import wave

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
MS = ROOT / 'diff/logs_omnidance/exp/MS'
CODE = ROOT / 'H3-posttrain/scripts_omnidance/exp/ms'
sys.path.insert(0, str(CODE))
import dance_caption_fusion as f
CORE = CODE / 'omni_dance_sft.py'
CORE_SHA = '4d82caffd4f6ce354fd6fbd240827d0cac69d66414de33e847501369b0e82dd2'
f.source_record(CORE, CORE_SHA)
spec = importlib.util.spec_from_file_location('omni_ms_style_full_frozen_core', CORE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
import prepare_omni_ms_style_full6500_v2 as labels


def full_data_boundary(args):
    if (args.data_scope != 'source_label_full' or args.smoke_steps != 0 or
            args.min_epochs != 3 or args.max_epochs != 10 or args.patience != 3 or args.relative_delta != .005):
        raise ValueError('Full source-label training requires explicit 6000/500, min3/max10, patience3/delta.005')
    rows, training, val = core.read_metadata(args.metadata)
    if (len(training), len(val)) != (6000, 500):
        raise ValueError('Full source-label training requires exactly 6000 train / 500 val')
    root = args.metadata.resolve().parent
    sidecar_source = f.source_record(root / 'source_label_sidecar.json')
    sidecar = f.read_json(sidecar_source['path'])
    if (sidecar.get('protocol') != labels.PROTOCOL or sidecar.get('complete') is not True or
            sidecar.get('authorization') != labels.AUTHORIZATION or sidecar.get('counts') != dict(train=6000, val=500) or
            sidecar.get('label_quality') != 'source_labels_unreviewed' or
            sidecar.get('semantic_approval') is not False or sidecar.get('human_review_completed') is not False or
            sidecar.get('original_source_window_mapping') != 'unknown' or
            any(sidecar.get(key, False) is not False for key in ('fixture_only', 'formal_reuse_forbidden'))):
        raise ValueError('Full labels require explicit weak supervision, never forged approval or fixture reuse')
    verified = {}
    def verify(source):
        actual = f.source_record(source['path'], source['sha256'])
        if actual != {key: source[key] for key in ('path', 'sha256', 'size_bytes')}:
            raise ValueError('Full source identity changed')
        verified[actual['path']] = actual
        return actual
    verify(sidecar_source)
    verify(sidecar['source_code'])
    verify(sidecar['frozen_trainer'])
    output = verify(sidecar['outputs']['omni.jsonl'])
    if Path(output['path']).resolve() != args.metadata.resolve():
        raise ValueError('Full metadata must match its declared output')
    for key in ('caps', 'split'):
        verify(sidecar['source_inputs'][key])
    original_caps = f.read_json(sidecar['source_inputs']['caps']['path'])
    original_split = f.read_json(sidecar['source_inputs']['split']['path'])
    if [row['id'] for row in rows] != original_split['train'] + original_split['val']:
        raise ValueError('Full data must preserve original split membership and order')
    version = sidecar['caption_version']
    records = sidecar['records']
    if len(records) != 6500 or [record['id'] for record in records] != [row['id'] for row in rows]:
        raise ValueError('Full source records and metadata membership differ')
    expected = []
    def check(pair):
        row, record = pair
        if (record['id'] != row['id'] or record['split'] != row['split'] or
                record['source_group'] != original_split['source_group_ids'][row['id']] or
                record['source_selected_caption'] != original_caps[row['id']] or row.get('caption_version') != version):
            raise ValueError('Full caption/source/group/split/version identity differs')
        ms = verify(record['ms'])
        audio = verify(record['audio'])
        original = f.read_json(ms['path'])
        if original['sid'] != row['id'] or f.object_sha(original) != record['raw_original_ms_sha256']:
            raise ValueError('Full original MotionScript identity differs')
        caption = labels.full_caption(original, record['source_selected_caption'])
        if f.object_sha(caption) != record['caption_sha256']:
            raise ValueError('Full caption differs from its documented deterministic normalization')
        with wave.open(audio['path'], 'rb') as stream:
            end = min(5.0625, stream.getnframes() / stream.getframerate())
            if stream.getnchannels() not in (1, 2) or not math.isfinite(end) or end < 5.0618:
                raise ValueError('Unexpected natural music excerpt window')
        window = [0.0, end]
        if record['audio_window_seconds'] != window:
            raise ValueError('Full natural audio window differs from source WAV')
        expected_row = dict(id=row['id'], source_group=record['source_group'], split=record['split'],
            caption_version=version, audios=[audio['path']], audio_window_seconds=window,
            messages=[dict(role='user', content=f.SFT_INSTRUCTION + '\n<audio>'), dict(role='assistant', content=caption)])
        if row != expected_row:
            raise ValueError('Full input/target fields drifted from the original source labels')
        return expected_row
    with ThreadPoolExecutor(max_workers=8) as executor:
        expected = list(executor.map(check, zip(rows, records)))
    if rows != expected:
        raise ValueError('Full source-label reconstruction failed')
    f.source_record(sidecar_source['path'], sidecar_source['sha256'])
    f.source_record(CORE, CORE_SHA)
    return rows, training, val, dict(protocol=labels.PROTOCOL, data_scope='source_label_full',
        metadata=output, sidecar=sidecar_source, full_entry_source=f.source_record(__file__),
        label_builder_source=f.source_record(labels.__file__), frozen_core_source=f.source_record(CORE, CORE_SHA),
        verified_sources=[verified[path] for path in sorted(verified)], caption_version=version,
        label_quality='source_labels_unreviewed', approval_chain_verified=False,
        semantic_approval=False, human_review_completed=False, original_source_window_mapping='unknown',
        train_ids=[row['id'] for row in training], val_ids=[row['id'] for row in val])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'validate'])
    parser.add_argument('--metadata', type=Path, default=MS / 'omni_ms_style_full6500_v2/omni.jsonl')
    parser.add_argument('--model', type=Path, default=Path('/opt/omni30b_dance_official_v1'))
    parser.add_argument('--model-id', default='Qwen/Qwen3-Omni-30B-A3B-Instruct@26291f793822fb6be9555850f06dfe95f2d7e695')
    parser.add_argument('--out', type=Path, default=Path('/opt/omni_ms_style_full_train_v2'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--distributed-timeout-seconds', type=int, default=7200)
    args = parser.parse_args()
    for key, value in dict(data_scope='source_label_full', smoke_steps=0, min_epochs=3, max_epochs=10,
            patience=3, relative_delta=.005, lora_rank=32, lora_alpha=64, lora_lr=5e-5,
            projector_lr=1e-5, accumulation=1, grad_clip=1., attention='sdpa', max_length=8192,
            val_generate=3, max_new_tokens=512, save_every=100, seed=42,
            zero_stage1=True, gradient_checkpointing=True, infer=False).items():
        setattr(args, key, value)
    if args.command == 'validate':
        _, train, val, identity = full_data_boundary(args)
        print(json.dumps(dict(counts={'train': len(train), 'val': len(val)}, data_provenance=identity,
            actual_training_started=False)), flush=True)
        return
    # Replace only this private module instance's data adapter; never mutate the on-disk
    # pilot entry or the module used by H3 source-label cache validation.
    core.validate_data_boundary = full_data_boundary
    initialize = core.dist.init_process_group
    core.dist.init_process_group = partial(initialize, timeout=timedelta(seconds=args.distributed_timeout_seconds))
    try:
        core.train(args)
    finally:
        core.dist.init_process_group = initialize
    f.source_record(CORE, CORE_SHA)


if __name__ == '__main__':
    main()
