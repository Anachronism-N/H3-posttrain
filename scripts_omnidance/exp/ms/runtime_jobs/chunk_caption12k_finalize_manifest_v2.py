import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def source(path):
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    probe = json.loads((OUT / 'selected_media_probe_cpu.json').read_text())
    assert probe['passed'] and probe['selected_clips'] == 14
    indexed = {row['id']: row for row in probe['records']}
    rows = [json.loads(line) for line in (OUT / 'manifest.jsonl').read_text().splitlines()]
    fixed = [json.loads(line) for line in (OUT / 'fixed3_manifest.jsonl').read_text().splitlines()]
    ids = (OUT.parent / 'ids_all.txt').read_text().split()
    assert len(rows) == len({row['id'] for row in rows}) == 12000
    assert [row['id'] for row in rows[:11992]] == ids
    assert len(fixed) == 3
    for row in rows + fixed:
        if row['dataset_role'] == 'extra8':
            assert not row['nlf_ready']
            row['nlf_path'] = str(OUT / 'nlf_extra8' / (row['id'] + '.npz'))
        if row['id'] in indexed:
            actual = indexed[row['id']]
            assert actual['video_path'] == row['video_path'] and actual['audio_path'] == row['audio_path'] and actual['source_group'] == row['source_group']
            row['protocol_actual_media_verified'] = True
            row['selected_media_probe'] = dict(path=str(OUT / 'selected_media_probe_cpu.json'), video=actual['video'], audio=actual['audio'], checks=actual['checks'], video_sha256=actual['video_sha256'], audio_sha256=actual['audio_sha256'], decoder=probe['decoder'])
    assert sum(row['protocol_actual_media_verified'] for row in rows) == 11
    assert all(row['protocol_actual_media_verified'] for row in fixed)
    assert all(row['split'] == 'train' for row in rows)
    assert len({row['source_group'] for row in rows}) == 12000
    assert all(row['normalized_audio_offset_seconds'] == 'unknown' for row in rows + fixed)
    for name, values in [('manifest.jsonl', rows), ('fixed3_manifest.jsonl', fixed), ('3smoke_manifest.jsonl', rows[:3])]:
        (OUT / name).write_text(''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in values))
    smoke_ids = [row['id'] for row in rows[:3]]
    (OUT / '3smoke_ids.txt').write_text('\n'.join(smoke_ids) + '\n')
    (OUT / 'nlf_extra8').mkdir(exist_ok=True)
    selection = json.loads((OUT / 'selection.json').read_text())
    selection['schema_revision'] = 2
    selection['extra8_nlf_output_directory'] = str(OUT / 'nlf_extra8')
    selection['authority_smoke_ids'] = smoke_ids
    selection['authority_smoke_manifest'] = str(OUT / '3smoke_manifest.jsonl')
    selection['selected_media_probe'] = source(OUT / 'selected_media_probe_cpu.json')
    selection['actual_media_verified_training_rows'] = 11
    selection['actual_media_verified_fixed3_rows'] = 3
    selection['actual_extra8_media_protocol_verified'] = True
    selection['selected_media_probe_decoder'] = probe['decoder']
    selection['same_protocol_claim'] = '12000 curated selected-metadata declarations; actual full CPU video decode and natural WAV coverage verified for all extra8, three authority smoke and fixed3 (14 total)'
    selection['media_hashes_precomputed'] = False
    selection['media_hashes_precomputed_scope'] = '14 selected clips only; video/audio SHA available in selected_media_probe_cpu.json; consumers still must verify actual consumed files'
    selection['existence_check_scope'] = 'all extra8, authority smoke3 and fixed3 decoded; authority NLF inventory complete; remaining authority video/audio checks deferred to consumption'
    selection['source_paths_hashes'][str(OUT / 'build_manifest.py')] = source(OUT / 'build_manifest.py')
    selection['source_paths_hashes'][str(Path(__file__).resolve())] = source(Path(__file__).resolve())
    selection['source_paths_hashes'][str(OUT / 'probe_selected_media_cpu.py')] = source(OUT / 'probe_selected_media_cpu.py')
    selection['artifacts'].update({name: source(OUT / name) for name in ('manifest.jsonl', 'fixed3_manifest.jsonl', '3smoke_manifest.jsonl', '3smoke_ids.txt', 'selected_media_probe_cpu.json')})
    dump(OUT / 'selection.json', selection)
    verification = json.loads((OUT / 'verification_cpu.json').read_text())
    verification['schema_revision'] = 2
    verification['manifest_sha256'] = source(OUT / 'manifest.jsonl')['sha256']
    verification['selection_sha256'] = source(OUT / 'selection.json')['sha256']
    verification['fixed3_manifest_sha256'] = source(OUT / 'fixed3_manifest.jsonl')['sha256']
    verification['authority_smoke_ids'] = smoke_ids
    verification['extra8_nlf_output_directory'] = str(OUT / 'nlf_extra8')
    verification['selected_media_probe'] = source(OUT / 'selected_media_probe_cpu.json')
    verification['selected_video_full_cpu_decode_count'] = 14
    verification['extra8_actual_media_protocol_passed'] = True
    verification['actual_verified_training_rows'] = 11
    verification['actual_verified_fixed3_rows'] = 3
    verification['limitations'] = ['No full 12000 video re-decoding; 14 clips (extra8, authority smoke3, fixed3) fully CPU-decoded; remaining selected CSV declarations only.', 'Only 14 video/audio SHA precomputed; worker must verify actual consumed video/audio/NLF before captioning.', 'Existing NLF fully inventoried, but contents CPU-smoked for three authority and three fixed cases.', 'All original-source/audio offset relationships remain unknown.', 'This is selection/input completeness, not captions, semantic review, or a new train-val split.']
    dump(OUT / 'verification_cpu.json', verification)
    print(json.dumps(dict(passed=True, manifest_sha256=verification['manifest_sha256'], selection_sha256=verification['selection_sha256'], fixed3_manifest_sha256=verification['fixed3_manifest_sha256'], smoke_ids=smoke_ids, extra8_media_probe_passed=True, verified_clips=14, pending_nlf_directory=str(OUT / 'nlf_extra8')), ensure_ascii=False))


if __name__ == '__main__':
    main()
