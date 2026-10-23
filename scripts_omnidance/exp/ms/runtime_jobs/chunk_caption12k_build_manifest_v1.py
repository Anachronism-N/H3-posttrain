import collections
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
MS = ROOT / 'diff/logs_omnidance/exp/MS'
OUT = MS / 'chunk_caption12k_input_v1'
DATA = ROOT / 'OmniDance_processed/pose_preprocess_outputs/mainline/normalized_704x1280/main'
SPLITS = ROOT / 'OmniDance_processed/pose_preprocess_outputs/mainline/vlm_filter_235b/splits'
ID_SOURCE = MS / 'ids_all.txt'
FIXED_SOURCE = MS / 'eval3_ids.txt'
PLANNER_SOURCES = [MS / 'music_direct6k_v2/split.json', MS / 'omni_ms_style_full6500_v3/declared_split.json']
PLANNER_REPORT = MS / 'music_direct6k_v2/selection_report.json'


def source(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=digest.hexdigest())


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    ids = ID_SOURCE.read_text().split()
    assert len(ids) == len(set(ids)) == 11992
    assert all(re.fullmatch(r'[A-Za-z0-9]+', sid) for sid in ids)
    authority = set(ids)
    fixed = FIXED_SOURCE.read_text().split()
    assert len(fixed) == len(set(fixed)) == 3 and not (set(fixed) & authority)
    planners = [json.loads(path.read_text()) for path in PLANNER_SOURCES]
    report = json.loads(PLANNER_REPORT.read_text())
    blocked = set(fixed) | set(report['blocked_ids']) | set(report['blocked_known_ids'])
    planner_val_ids = set()
    planner_val_groups = set()
    planner_group_maps = {}
    for path, planner in zip(PLANNER_SOURCES, planners):
        planner_val_ids.update(planner['val'])
        planner_val_groups.update(planner['source_group_ids'][sid] for sid in planner['val'])
        planner_group_maps[str(path)] = planner['source_group_ids']
    blocked.update(planner_val_ids)
    blocked.update(planner_val_groups)
    selected_rows = {}
    candidates = []
    known_sources = {}
    counts = collections.Counter()
    csv_headers = {}
    authority_sources = set()
    for split in ('train', 'val', 'test'):
        path = SPLITS / (split + '.csv')
        with path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.reader(stream)
            header = next(reader)
            csv_headers[split] = header
            stem_indices = [i for i, name in enumerate(header) if name == 'video_stem']
            for line_number, values in enumerate(reader, 2):
                assert len(values) == len(header), (path, line_number)
                assert len({values[i] for i in stem_indices}) == 1, (path, line_number, 'duplicate stem columns disagree')
                row = dict(zip(header, values))
                sid = row['video_stem']
                counts[split + '_csv_rows'] += 1
                if sid in authority or sid in blocked or (split == 'train' and len(candidates) < 1000):
                    previous = known_sources.get(sid)
                    assert previous is None or previous == row['source_video'], ('source mismatch', sid)
                    known_sources[sid] = row['source_video']
                stored = dict(split=split, csv_line_number=line_number, values=values, row=row, csv_path=str(path))
                if sid in authority or sid in fixed:
                    assert sid not in selected_rows, ('duplicate selected sid', sid)
                    selected_rows[sid] = stored
                if sid in authority:
                    counts[split + '_authority_matches'] += 1
                    authority_sources.add(row['source_video'])
                if split == 'train' and sid not in authority and len(candidates) < 1000:
                    candidates.append(stored)
    assert set(ids + fixed) <= set(selected_rows)
    assert counts['train_authority_matches'] == 11992
    assert counts['val_authority_matches'] == counts['test_authority_matches'] == 0
    # Metadata lookup only resolves held-out IDs absent from curated CSVs.
    missing_blocked_sources = []
    for sid in sorted(blocked - set(known_sources)):
        path = DATA / 'meta' / (sid + '.json')
        if path.is_file():
            item = json.loads(path.read_text())
            if item.get('source_video'):
                known_sources[sid] = item['source_video']
        if sid not in known_sources:
            missing_blocked_sources.append(sid)
    blocked_sources = {known_sources[sid] for sid in blocked if sid in known_sources}
    nlf_dirs = [MS / name for name in ('nlf', 'nlf_recovery', 'nlf_recovery_v2', 'nlf_eval3')]
    nlf_inventory = {}
    nlf_by_id = collections.defaultdict(list)
    for directory in nlf_dirs:
        files = sorted(directory.glob('*.npz'))
        nlf_inventory[str(directory)] = dict(files=len(files), authority_covered=len({p.stem for p in files} & authority), nonauthority_ids=[p.stem for p in files if p.stem not in authority])
        for path in files:
            nlf_by_id[path.stem].append(path)
    assert all((MS / 'nlf' / (sid + '.npz')) in nlf_by_id[sid] for sid in ids)
    extra = []
    extra_sources = set()
    for stored in candidates:
        row = stored['row']
        sid, origin = row['video_stem'], row['source_video']
        if sid in blocked or not origin or origin in blocked_sources or origin in authority_sources or origin in extra_sources:
            continue
        if any(mapping.get(sid) in planner_val_groups for mapping in planner_group_maps.values()):
            continue
        expected = ('704', '1280', '16', '81', 'pass', '1', 'ok')
        actual = tuple(row[k] for k in ('target_width', 'target_height', 'target_fps', 'target_frames', 'category', 'overall_pass', 'status'))
        if actual != expected or not all(Path(row[k]).is_file() for k in ('video', 'input_audio', 'meta')):
            continue
        extra.append(stored)
        extra_sources.add(origin)
        if len(extra) == 8:
            break
    assert len(extra) == len(extra_sources) == 8
    extra_ids = [stored['row']['video_stem'] for stored in extra]
    ordered = [selected_rows[sid] for sid in ids] + extra
    assert len(ordered) == 12000
    sources = {str(path): source(path) for path in [ID_SOURCE, FIXED_SOURCE, *PLANNER_SOURCES, PLANNER_REPORT, Path(__file__), *[SPLITS / (split + '.csv') for split in ('train', 'val', 'test')]]}
    protocol_counts = collections.Counter()
    original_offsets_known = 0
    def record(stored, role):
        nonlocal original_offsets_known
        row = stored['row']
        sid = row['video_stem']
        if role == 'authority':
            nlf = MS / 'nlf' / (sid + '.npz')
        elif role == 'fixed3':
            nlf = MS / 'nlf_eval3' / (sid + '.npz')
        else:
            available = nlf_by_id.get(sid, [])
            nlf = available[0] if available else OUT / 'nlf_extra8' / (sid + '.npz')
        nlf_ready = nlf in nlf_by_id.get(sid, [])
        def numeric(field):
            value = row.get(field, '').strip()
            try:
                return float(value) if value else 'unknown'
            except ValueError:
                return 'unknown'
        selected_start = numeric('selected_start_frame')
        source_fps = numeric('source_fps')
        source_start = selected_start / source_fps if isinstance(selected_start, float) and isinstance(source_fps, float) and source_fps > 0 else 'unknown'
        if role != 'fixed3':
            original_offsets_known += source_start != 'unknown'
            protocol_counts['x'.join(row[k] for k in ('target_width', 'target_height', 'target_fps', 'target_frames'))] += 1
        output = dict(id=sid, split=stored['split'], dataset_role=role, video_path=row['video'], audio_path=row['input_audio'], nlf_path=str(nlf), source_group=row['source_video'], source_group_basis='exact original source_video path from curated CSV; no claim of content or audio duplicate isolation', source_video=row['source_video'], selected_metadata_path=row['meta'], selected_metadata_source='curated CSV raw row; per-clip JSON not fully read', selected_metadata={k: row[k] for k in ('source_row_index', 'duration_seconds', 'fps', 'source_fps', 'selected_start_frame', 'natural_end_frame_exclusive', 'head_trim_frames', 'tail_bad_frames', 'natural_output_frames', 'tail_padding_frames', 'target_width', 'target_height', 'target_fps', 'target_frames', 'status', 'category', 'overall_pass')}, source_start_seconds_declared=source_start, video_start_seconds=0.0, video_end_seconds=5.0625, normalized_audio_offset_seconds='unknown', original_source_audio_offset_seconds='unknown', original_source_window_mapping='unknown', declared_protocol=dict(width=704, height=1280, fps=16, frames=81, start_seconds=0.0, end_seconds=5.0625), protocol_actual_media_verified=False, nlf_ready=nlf_ready, nlf_status='cached' if nlf_ready else 'pending_extra8_gpu_extraction', media_sha256_status='pending_worker_consumption', source_csv=str(stored['csv_path']), source_csv_line_number=stored['csv_line_number'], source_csv_raw_values=stored['values'], source_csv_sha256=sources[stored['csv_path']]['sha256'], planner_source_groups={path: mapping.get(sid, 'not_in_planner') for path, mapping in planner_group_maps.items()}, label_quality='unreviewed_inputs_only', caption_completed=False, human_review_completed=False, fixture_only=False)
        if role == 'fixed3':
            output['excluded_from_training_input'] = True
            output['media_path_exists_at_build'] = {name: Path(output[name]).is_file() for name in ('video_path', 'audio_path', 'nlf_path', 'selected_metadata_path')}
        elif role == 'extra8':
            output['media_path_exists_at_build'] = {name: Path(output[name]).is_file() for name in ('video_path', 'audio_path', 'selected_metadata_path')}
        return output
    records = [record(stored, 'authority' if index < 11992 else 'extra8') for index, stored in enumerate(ordered)]
    fixed_records = [record(selected_rows[sid], 'fixed3') for sid in fixed]
    assert all(row['nlf_ready'] for row in fixed_records)
    for name, rows in [('manifest.jsonl', records), ('fixed3_manifest.jsonl', fixed_records)]:
        with (OUT / name).open('x', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    pending = [row['id'] for row in records if not row['nlf_ready']]
    for name, values in [('ids.txt', ids + extra_ids), ('extra8_ids.txt', extra_ids), ('pending_nlf_ids.txt', pending), ('fixed3_ids.txt', fixed)]:
        with (OUT / name).open('x') as stream:
            stream.write('\n'.join(values) + ('\n' if values else ''))
    authority_planner_val_overlap = sorted(authority & planner_val_ids)
    summary = dict(schema='chunk-caption12k-input-manifest-v1', complete_input_selection=True, caption_completed=False, selection_count=12000, authority_count=11992, extra_count=8, unique_ids=12000, curated_split_counts=dict(collections.Counter(row['split'] for row in records)), curated_csv_counts=dict(counts), csv_headers=csv_headers, duplicate_csv_headers={split: {key: number for key, number in collections.Counter(header).items() if number > 1} for split, header in csv_headers.items()}, source_paths_hashes=sources, authority_input=str(ID_SOURCE), prohibited_selection_input=str(MS.parent / 'H3AR/full12k_ids.txt'), prohibited_selection_input_used=False, authority_nlf_directory=str(MS / 'nlf'), nonexistent_alias_not_used=str(MS / 'nlf11992'), nlf_inventory=nlf_inventory, cached_nlf_count=sum(row['nlf_ready'] for row in records), pending_nlf_count=len(pending), pending_nlf_ids=pending, extra8_ids=extra_ids, extra8_source_groups=sorted(extra_sources), extra8_selection_order='first eligible curated train CSV rows, prefer already inventoried cached NLF when present', extra8_planner_val_overlap=[], extra8_fixed3_overlap=[], extra8_authority_source_overlap=[], extra8_distinct_source_groups=True, planner_validation_union_count=len(planner_val_ids), planner_validation_group_union_count=len(planner_val_groups), fixed3_ids=fixed, fixed3_manifest=str(OUT / 'fixed3_manifest.jsonl'), authority_planner_validation_overlap_count=len(authority_planner_val_overlap), authority_planner_validation_overlap_ids=authority_planner_val_overlap, authority_membership_preserved=True, authority_old_planner_val_overlap_policy='This independent annotation pool is all curated train. Historical internal planner val membership is disclosed, not removed from authoritative 11992; this is not a new training split.', blocked_source_unresolved_ids=missing_blocked_sources, source_group_count=len({row['source_group'] for row in records}), declared_metadata_protocol_counts=dict(protocol_counts), declared_source_start_offsets_known_count=original_offsets_known, declared_source_start_offsets_unknown_count=12000 - original_offsets_known, audio_offsets_unknown_count=12000, raw_csv_preservation='source_csv_raw_values aligned to csv_headers[split], preserving duplicate headers without loss', media_hashes_precomputed=False, media_hash_policy='consumer must hash actual video/audio/NLF at consumption and validate matching observed span; no full-media hash prerequisite for handing off selection', all_media_existence_checked=False, existence_check_scope='extra8 and fixed3 paths only; authority NLF membership is fully inventoried; authority video/audio validation deferred to worker', same_protocol_claim='curated selected metadata declaration, not full media re-decoding', semantic_quality_claim=False, human_review_completed=False, gpu_used=False, source_files_modified=False)
    summary['artifacts'] = {name: source(OUT / name) for name in ('manifest.jsonl', 'fixed3_manifest.jsonl', 'ids.txt', 'extra8_ids.txt', 'pending_nlf_ids.txt', 'fixed3_ids.txt')}
    write_json(OUT / 'selection.json', summary)
    print(json.dumps({key: summary[key] for key in ('selection_count', 'authority_count', 'extra_count', 'cached_nlf_count', 'pending_nlf_count', 'curated_split_counts', 'declared_metadata_protocol_counts', 'source_group_count', 'authority_planner_validation_overlap_count', 'blocked_source_unresolved_ids', 'extra8_ids')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
