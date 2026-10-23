"""CPU source-identity admission for authorized MS/style weak full SFT, not fusion review."""
import ast
import copy
import json
from pathlib import Path
import re
import types
import wave

import dance_caption_fusion as fusion

PROTOCOL = 'omni-source-label-full-exact-window-replacement-v2'
AUTHORIZATION = ('User authorized original MS plus original style weak supervision on 2026-10-23; '
                 'no caption-LLM or individual human review prerequisite')
COUNTS = {'train': 6000, 'val': 500}
WINDOW = [0., 5.0625]
STATUS = 'admitted_source_label_full'
FALSE_FLAGS = ('human_review_completed', 'semantic_approval', 'fixture_only', 'formal_reuse_forbidden')


def _weak_flags(value, approval=False):
    if (value.get('label_quality') != 'source_labels_unreviewed' or
            value.get('original_source_window_mapping') != 'unknown' or
            any(value.get(key) is not False for key in FALSE_FLAGS) or
            approval and value.get('approval_chain_verified') is not False):
        raise ValueError('Weak full requires explicit unreviewed/unknown-window/false boolean identity')


def _counts(value, expected, name):
    if value != expected or not isinstance(value, dict) or any(type(v) is not int for v in value.values()):
        raise ValueError('Full exact counts differ: ' + name)


def _replacement_genre(record, lineage, choice, parent, candidate_source, original_source):
    genre, original_genre = candidate_source['dims']['genre'], parent['genre']
    matched = genre == original_genre
    if original_source['dims']['genre'] != original_genre:
        raise ValueError('Full original failed genre differs from original source pool')
    if any(value.get('genre' if value is not lineage else 'style_label') != genre or
           value.get('original_genre') != original_genre or value.get('genre_match') is not matched
           for value in (record, lineage, choice)):
        raise ValueError('Full replacement truthful genre/original genre/boolean lineage differs')
    rule = record['split_admission_rule']
    reserve = ('original_heldout_rank_unused_group_reserve' if record['split'] == 'val'
               else 'original_train_eligible_unused_group_reserve')
    if matched:
        if rule not in ('same_original_partition_group', reserve):
            raise ValueError('Full matching genre must use the declared original-partition reserve rule')
    elif (rule != reserve + '_same_coarse_genre_shortfall' or
          not isinstance(original_source['dims'].get('coarse'), str) or
          not original_source['dims']['coarse'] or
          candidate_source['dims'].get('coarse') != original_source['dims']['coarse']):
        raise ValueError('Full truthful genre scarcity replacement must retain original coarse and explicit rule')


def _labeler(path):
    tree = ast.parse(Path(path).read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'source_label_caption']
    if len(functions) != 1:
        raise ValueError('Missing unique training caption implementation')
    module = types.ModuleType('source_label_cpu_boundary')
    module.re = re
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), module.__dict__)
    return module.source_label_caption


def verify_collection(metadata_path, caption_version, rows, training_rows, val_rows, *, require_admitted=True):
    """Validate actual sources once; offline pending verification never permits train admission."""
    metadata_path = Path(metadata_path).absolute()
    root = metadata_path.parent
    verified, snapshots = {}, {}
    def verify(source, *, partial=False):
        if (not isinstance(source, dict) or not isinstance(source.get('path'), str) or
                not re.fullmatch('[0-9a-f]{64}', str(source.get('sha256', ''))) or
                not Path(source['path']).is_absolute()):
            raise ValueError('Invalid declared full source path/SHA')
        path = source['path']
        if path not in verified:
            verified[path] = fusion.source_record(path, source['sha256'])
            stat = Path(path).stat()
            snapshots[path] = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
        actual = verified[path]
        if actual['sha256'] != source['sha256'] or (not partial or 'size_bytes' in source) and (
                type(source.get('size_bytes')) is not int or actual['size_bytes'] != source['size_bytes']):
            raise ValueError('Declared full source identity differs: ' + path)
        return actual
    def source_at(path):
        actual = fusion.source_record(path)
        verified[actual['path']] = actual
        stat = Path(actual['path']).stat()
        snapshots[actual['path']] = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
        return actual
    def sources_nested(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                verify(value, partial=True)
            else:
                for item in value.values():
                    sources_nested(item)
        elif isinstance(value, list):
            for item in value:
                sources_nested(item)
    sidecar_source = source_at(root / 'source_label_sidecar.json')
    sidecar = fusion.read_json(sidecar_source['path'])
    _weak_flags(sidecar, approval=require_admitted)
    if sidecar.get('protocol') != PROTOCOL or sidecar.get('complete') is not True or sidecar.get('collection_complete') is not True:
        raise ValueError('Full collection must have the explicit v2 complete protocol')
    if sidecar.get('authorization') != AUTHORIZATION:
        raise ValueError('Full weak supervision authorization differs')
    if require_admitted and (sidecar.get('admission_status') != STATUS or
            sidecar.get('admitted') is not True or sidecar.get('ready_for_training') is not True):
        raise ValueError('Full sidecar is pending or not explicitly admitted for source_label_full')
    manifest_source = verify(sidecar['manifest'])
    lineage_source = verify(sidecar['lineage'])
    manifest = fusion.read_json(manifest_source['path'])
    lineage = fusion.read_json(lineage_source['path'])
    _weak_flags(manifest, approval=require_admitted)
    _weak_flags(lineage)
    if manifest.get('protocol') != PROTOCOL or lineage.get('protocol') != PROTOCOL:
        raise ValueError('Full manifest/lineage protocol differs')
    if manifest.get('complete') is not True or manifest.get('collection_complete') is not True or manifest.get('blocked') is not False:
        raise ValueError('Full manifest is incomplete or blocked')
    if manifest.get('authorization') != AUTHORIZATION:
        raise ValueError('Full manifest authorization differs')
    if require_admitted and (manifest.get('admission_status') != STATUS or
            manifest.get('admitted') is not True or manifest.get('ready_for_training') is not True):
        raise ValueError('Full manifest is pending or not explicitly admitted')
    for value in (sidecar, manifest):
        for key in ('counts', 'success_counts', 'generated_caption_counts'):
            _counts(value.get(key), COUNTS, key)
        _counts(value.get('failed_counts'), {'train': 0, 'val': 0}, 'failed_counts')
    _counts({'train': len(training_rows), 'val': len(val_rows)}, COUNTS, 'actual reader')
    for declared in (sidecar['source_code'], manifest['source_code'], sidecar['trainer_source'], manifest['trainer_source']):
        verify(declared)
    training_entry = source_at(Path(__file__).with_name('omni_dance_full_sft.py'))
    validator_source = source_at(__file__)
    if require_admitted and (verify(sidecar['training_entry']) != training_entry or
                            verify(sidecar['admission_validator']) != validator_source):
        raise ValueError('Full admission must bind the actual new entry and validator, not the frozen pilot')
    labeler = _labeler(training_entry['path'])
    helper_source = source_at(fusion.__file__)
    outputs = {name: verify(source) for name, source in sidecar['outputs'].items()}
    if outputs['omni.jsonl']['path'] != str(metadata_path):
        raise ValueError('Full metadata is not the declared omni.jsonl')
    for key, name in (('manifest', 'source_label_manifest.json'), ('lineage', 'source_label_lineage.json'),
                      ('frozen_manifest', 'frozen_source_manifest.json'), ('declared_split', 'declared_split.json'),
                      ('replacement_report', 'replacement_report.json')):
        if verify(sidecar[key]) != outputs[name]:
            raise ValueError('Full sidecar declared source/output identity differs: ' + key)
    frozen = fusion.read_json(outputs['frozen_source_manifest.json']['path'])
    _weak_flags(frozen)
    freeze = manifest['fixed_source_inputs']
    fixed_sha = fusion.object_sha(freeze)
    sources_nested(freeze)
    if (frozen.get('complete') is not True or frozen.get('protocol') != PROTOCOL or
            frozen.get('frozen_inputs') != freeze or frozen.get('fixed_source_inputs_sha256') != fixed_sha or
            frozen.get('caption_version') != caption_version or
            caption_version != PROTOCOL + ':' + fixed_sha or
            any(value.get('caption_version') != caption_version or value.get('fixed_source_inputs_sha256') != fixed_sha
                for value in (sidecar, manifest)) or lineage.get('source_inputs_sha256') != fixed_sha):
        raise ValueError('Full single caption version/frozen inputs identity differs')
    _counts(frozen.get('counts'), COUNTS, 'frozen counts')
    if sidecar['source_inputs'] != manifest['source_inputs'] or manifest['source_inputs'] != freeze['original_source_inputs']:
        raise ValueError('Full original input declarations differ')
    sources_nested(sidecar['source_inputs'])
    parent = fusion.read_json(verify(freeze['parent_sources']['source_label_manifest.json'])['path'])
    original_split = fusion.read_json(verify(freeze['original_source_inputs']['split'])['path'])
    selected_caps = fusion.read_json(verify(freeze['original_source_inputs']['caps'])['path'])
    report = fusion.read_json(verify(freeze['original_source_inputs']['selection_report'])['path'])
    refs = fusion.read_json(verify(freeze['original_source_inputs']['references'])['path'])['records']
    parent_records = parent['records']
    _counts(parent.get('counts'), COUNTS, 'original counts')
    _counts({p: len(original_split[p]) for p in COUNTS}, COUNTS, 'original split')
    if [r['id'] for r in parent_records] != original_split['train'] + original_split['val']:
        raise ValueError('Original parent order/split differs')
    old = {r['id']: r for r in parent_records}
    replacements = fusion.read_json(outputs['replacement_report.json']['path'])
    _weak_flags(replacements)
    _counts(replacements.get('counts'), {'train': 87, 'val': 7}, 'replacement counts')
    _counts(manifest.get('replacement_counts'), {'train': 87, 'val': 7}, 'manifest replacement counts')
    choices = replacements['records']
    if (replacements.get('complete') is not True or replacements.get('protocol') != PROTOCOL or
            choices != freeze['replacements'] or len(choices) != 94 or
            {r['original_id'] for r in choices} != {r['id'] for r in parent_records if r['blocked']}):
        raise ValueError('Replacement lineage differs from exact original failed members')
    pool_source = verify(freeze['original_selection_source_inputs']['pool'], partial=True)
    pool_report_source = verify(freeze['original_selection_source_inputs']['pool_report'], partial=True)
    pool = fusion.read_json(pool_source['path'])
    pool_report = fusion.read_json(pool_report_source['path'])
    choice_map = {r['replacement_id']: r for r in choices}
    if len(choice_map) != 94:
        raise ValueError('Duplicate replacement IDs')
    split = fusion.read_json(outputs['declared_split.json']['path'])
    if split.get('protocol') != PROTOCOL or split.get('replacement_lineage') != choices or split.get('original_split') != freeze['original_source_inputs']['split']:
        raise ValueError('Declared full split lineage differs')
    records = manifest['records']
    if records != sidecar['records'] or len(records) != 6500 or len(lineage['records']) != 6500:
        raise ValueError('Full sidecar/manifest/lineage records differ')
    original_groups = {p: {r['source_group'] for r in parent_records if r['split'] == p} for p in COUNTS}
    groups = {p: set() for p in COUNTS}
    audio_groups = {p: {'path': set(), 'sha256': set()} for p in COUNTS}
    expected, ids = [], set()
    for record, lin, parent_record in zip(records, lineage['records'], parent_records):
        _weak_flags(record); _weak_flags(lin)
        sid, partition = record['id'], record['split']
        fusion.valid_id(sid)
        if sid in ids or partition not in COUNTS or partition != parent_record['split']:
            raise ValueError('Full duplicate ID or altered original slot partition')
        ids.add(sid)
        group = record.get('source_group')
        if (not isinstance(group, str) or not group or record.get('blocked') is not False or
                record.get('failure_reasons') != [] or record.get('caption_version') != caption_version or
                record.get('fixed_source_inputs_sha256') != fixed_sha or
                record.get('audio_window_seconds') != WINDOW):
            raise ValueError('Full record source group/status/version/window differs')
        groups[partition].add(group)
        ms_source = verify(record['provenance']['ms'])
        audio_source = verify(record['provenance']['audio'])
        if verify(record['declared_ms_source'], partial=True) != ms_source or verify(record['declared_audio_source'], partial=True) != audio_source:
            raise ValueError('Full declared/actual MS/audio sources differ')
        material = fusion.read_json(ms_source['path'])
        if material != record['source_MS'] or material.get('sid') != sid:
            raise ValueError('Full original MS source bytes/sid differ')
        caption = labeler(record)
        genre = record['source_selected_caption']['dims']['genre']
        if record['genre'] != genre or record.get('motion_caption') != caption or record.get('caption_sha256') != fusion.object_sha(caption):
            raise ValueError('Full original genre/caption source differs')
        if parent_record['blocked']:
            choice = choice_map[sid]
            pool_item = pool[sid]
            raw_motion = ' '.join(text.strip() for text in material['ms'] if text.strip())
            if (pool_item.get('audit_only') or pool_item['motion_original'] != raw_motion or
                    pool_item['dims']['genre'] != genre or
                    record['source_selected_caption'].get('teacher_original_motion') != raw_motion or
                    verify(pool_report['provenance'][sid]['ms'], partial=True) != ms_source):
                raise ValueError('Full replacement actual original pool MS/genre source differs')
            _replacement_genre(record, lin, choice, parent_record, pool_item, pool[parent_record['id']])
            if (choice['original_id'] != parent_record['id'] or choice['split'] != partition or
                    choice['genre'] != genre or
                    choice['replacement_source_group'] != group or choice['original_source_group'] != parent_record['source_group'] or
                    record.get('replacement_for') != parent_record['id'] or
                    record.get('split_admission_rule') != choice['split_admission_rule'] or
                    group in original_groups['val' if partition == 'train' else 'train'] or
                    record.get('curated_train_pass_verified') is not True or
                    verify(choice['ms_source']) != ms_source or verify(choice['audio_source']) != audio_source):
                raise ValueError('Full replacement source/same-partition/same-genre identity differs')
        else:
            previous = copy.deepcopy(record)
            reuse = previous.pop('reuse_provenance')
            previous['caption_version'] = parent_record['caption_version']
            previous['fixed_source_inputs_sha256'] = parent_record['fixed_source_inputs_sha256']
            if (previous != parent_record or sid != parent_record['id'] or
                    reuse.get('original_record_sha256') != fusion.object_sha(parent_record) or
                    verify(reuse['parent_manifest']) != verify(freeze['parent_sources']['source_label_manifest.json']) or
                    selected_caps.get(sid) != record['source_selected_caption'] or
                    original_split['source_group_ids'][sid] != group or refs[sid]['source_group'] != group or
                    refs[sid]['audio_path'] != audio_source['path'] or refs[sid]['audio_sha256'] != audio_source['sha256'] or
                    verify(report['artifacts'][sid]['ms'], partial=True) != ms_source or
                    verify(report['artifacts'][sid]['audio'], partial=True) != audio_source):
                raise ValueError('Full reused success differs from original frozen source/partition/caption')
        if record['audio_path'] != audio_source['path'] or record['audios'] != [audio_source['path']]:
            raise ValueError('Full audio path/audios differ')
        with wave.open(audio_source['path'], 'rb') as wav:
            rate, frames, channels, width = wav.getframerate(), wav.getnframes(), wav.getnchannels(), wav.getsampwidth()
            required = round(WINDOW[1] * rate)
            actual = len(wav.readframes(required)) // (channels * width)
            info = dict(sample_rate=rate, nframes=frames, channels=channels, sample_width_bytes=width,
                        compression_type=wav.getcomptype(), duration_seconds=frames / rate,
                        required_frames=required, actual_window_frames=actual, requested_window_seconds=WINDOW,
                        padding_applied=False, window_shortened=False)
            if (rate <= 0 or channels not in (1, 2) or wav.getcomptype() != 'NONE' or
                    frames / rate < WINDOW[1] or actual / rate < WINDOW[1] or record['audio_metadata'] != info):
                raise ValueError('Full actual exact 5.0625s PCM/header/declared window differs')
        for kind in audio_groups[partition]:
            audio_groups[partition][kind].add(audio_source[kind])
        if (lin['id'] != sid or lin['split'] != partition or lin['source_group'] != group or
                lin['raw_original_ms'] != material['ms'] or lin['style_label'] != genre or lin['caption'] != caption or
                lin['caption_sha256'] != fusion.object_sha(caption) or verify(lin['ms_source']) != ms_source or
                verify(lin['audio_source']) != audio_source or
                lin['source_selected_caption_sha256'] != fusion.object_sha(record['source_selected_caption']) or
                lin['fixed_source_inputs_sha256'] != fixed_sha):
            raise ValueError('Full caption/MS/genre/audio/split lineage differs')
        expected.append(dict(id=sid, source_group=group, split=partition, caption_version=caption_version,
            messages=[dict(role='user', content=fusion.SFT_INSTRUCTION + '\n<audio>'), dict(role='assistant', content=caption)],
            audios=[audio_source['path']], audio_window_seconds=WINDOW))
    if groups['train'] & groups['val'] or any(audio_groups['train'][k] & audio_groups['val'][k] for k in ('path', 'sha256')):
        raise ValueError('Full source group/audio path/audio SHA leaks across splits')
    if manifest.get('cross_split_audio') != {'path': [], 'sha256': []} or manifest.get('cross_split_source_groups') != [] or manifest.get('source_group_split_disjoint') is not True:
        raise ValueError('Full declared disjoint source/media audit differs')
    def raw_rows(path):
        return [json.loads(line, object_pairs_hook=fusion.unique_object) for line in Path(path).read_text().splitlines() if line.strip()]
    if rows != expected or raw_rows(metadata_path) != expected:
        raise ValueError('Full actual Omni rows differ from exact original MS/style export')
    for partition, actual in (('train', training_rows), ('val', val_rows)):
        selected = [r for r in expected if r['split'] == partition]
        if actual != selected or raw_rows(outputs[partition + '.jsonl']['path']) != selected or split[partition] != [r['id'] for r in selected]:
            raise ValueError('Full actual declared split/partition rows differ')
    if split['source_group_ids'] != {r['id']: r['source_group'] for r in expected}:
        raise ValueError('Full declared split groups differ')
    for path, snapshot in snapshots.items():
        stat = Path(path).stat()
        if snapshot != (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino):
            raise ValueError('Full source changed during admission: ' + path)
    return dict(protocol=PROTOCOL, data_scope='source_label_full', metadata=outputs['omni.jsonl'],
        sidecar=sidecar_source, manifest=manifest_source, lineage=lineage_source, training_entry=training_entry,
        validator_source=validator_source, label_helper_source=helper_source,
        all_verified_sources=sorted(verified.values(), key=lambda r: r['path']), caption_version=caption_version,
        authorization=AUTHORIZATION, label_quality='source_labels_unreviewed', approval_chain_verified=False,
        human_review_completed=False, semantic_approval=False, original_source_window_mapping='unknown',
        fixture_only=False, formal_reuse_forbidden=False, actual_source_identity_verified=True,
        admission_status=STATUS if require_admitted else 'offline_collection_verified_not_admitted',
        admitted=require_admitted, train_ids=[r['id'] for r in training_rows], val_ids=[r['id'] for r in val_rows])


def source_label_full_provenance(metadata_path, caption_version, rows, training_rows, val_rows):
    return verify_collection(metadata_path, caption_version, rows, training_rows, val_rows, require_admitted=True)
