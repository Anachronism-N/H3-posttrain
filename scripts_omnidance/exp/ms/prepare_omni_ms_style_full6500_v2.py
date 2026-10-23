import copy
import json
from pathlib import Path
import re
import sys
import wave

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
MS = ROOT / 'diff/logs_omnidance/exp/MS'
sys.path.insert(0, str(ROOT / 'H3-posttrain/scripts_omnidance/exp/ms'))
import dance_caption_fusion as f
import omni_dance_sft as omni

OUT = MS / 'omni_ms_style_full6500_v2'
PARENT = MS / 'omni_ms_style_full6500_v1/source_label_manifest.json'
PROTOCOL = 'omni-source-label-full-natural-window-v2'
AUTHORIZATION = 'User authorized reuse of existing MotionScript + style for full6000/500 experiments on 2026-10-23'


def full_caption(source_ms, selected_caption):
    material = copy.deepcopy(source_ms)
    material['ms'] = [re.sub(r'\bface\s+up\b', 'supine', text, flags=re.I) for text in material['ms']]
    return omni.source_label_caption(dict(source_MS=material, source_selected_caption=selected_caption))


def main():
    if not MS.is_dir() or OUT.exists():
        raise ValueError('Require an existing parent and fresh full-data output')
    trainer = f.source_record(omni.__file__, '4d82caffd4f6ce354fd6fbd240827d0cac69d66414de33e847501369b0e82dd2')
    parent_source = f.source_record(PARENT)
    parent = f.read_json(PARENT)
    if parent['counts'] != {'train': 6000, 'val': 500} or len(parent['records']) != 6500:
        raise ValueError('Original full source split changed')
    version = PROTOCOL + ':' + f.object_sha(dict(parent=parent_source, normalization='face up -> supine; exact available audio duration, no padding'))
    rows, records = [], []
    for record in parent['records']:
        ms, audio = record['provenance']['ms'], record['provenance']['audio']
        if any(item['stage'] not in ('audio_window', 'source_label_caption') for item in record['failure_reasons']):
            raise ValueError('Source identity failure cannot be normalized')
        original = record['source_MS']
        caption = full_caption(original, record['source_selected_caption'])
        info = record['audio_metadata']
        end = min(5.0625, info['nframes'] / info['sample_rate'])
        if end < 5.0618 or info['compression_type'] != 'NONE':
            raise ValueError('Unexpected audio shortening; do not silently accept a different window')
        rows.append(dict(id=record['id'], source_group=record['source_group'], split=record['split'],
            caption_version=version, audios=[audio['path']], audio_window_seconds=[0.0, end],
            messages=[dict(role='user', content=f.SFT_INSTRUCTION + '\n<audio>'), dict(role='assistant', content=caption)]))
        records.append(dict(id=record['id'], source_group=record['source_group'], split=record['split'],
            audio=audio, ms=ms, style_label=record['genre'], audio_window_seconds=[0.0, end],
            caption_sha256=f.object_sha(caption), source_selected_caption=record['source_selected_caption'],
            normalization='Only body orientation face up is rendered supine; actual audio endpoint retained.',
            raw_original_ms_sha256=f.object_sha(original), semantic_approval=False))
    if len(rows) != 6500 or {k: sum(row['split'] == k for row in rows) for k in ('train', 'val')} != parent['counts']:
        raise ValueError('Exact full source membership/counts changed')
    f.source_record(omni.__file__, trainer['sha256'])
    f.source_record(PARENT, parent_source['sha256'])
    OUT.mkdir(exist_ok=False)
    f.atomic_new(OUT / 'omni.jsonl', ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    f.atomic_json(OUT / 'source_label_sidecar.json', dict(protocol=PROTOCOL, complete=True,
        counts=parent['counts'], caption_version=version, authorization=AUTHORIZATION,
        label_quality='source_labels_unreviewed', semantic_approval=False, human_review_completed=False,
        original_source_window_mapping='unknown', source_code=f.source_record(__file__), frozen_trainer=trainer,
        parent_manifest=parent_source, source_inputs=parent['source_inputs'], records=records,
        outputs={'omni.jsonl': f.source_record(OUT / 'omni.jsonl')},
        audio_policy='Use min(5.0625, exact original WAV duration); maximum shortening 16 samples at24k, never padding or forged full length.',
        orientation_policy='face up -> supine is body orientation, not an identity description',
        fixture_only=False, formal_reuse_forbidden=False))
    _, train, val = omni.read_metadata(OUT / 'omni.jsonl')
    assert (len(train), len(val)) == (6000, 500)
    print(json.dumps(dict(out=str(OUT), counts=parent['counts'],
        exact_natural_audio_windows=sum(row['audio_window_seconds'][1] < 5.0625 for row in rows),
        metadata=f.source_record(OUT / 'omni.jsonl'), sidecar=f.source_record(OUT / 'source_label_sidecar.json'),
        semantic_approval=False, student_training_started=False)), flush=True)


if __name__ == '__main__':
    main()
