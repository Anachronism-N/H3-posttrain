"""Finite experiment completion checks; never equate partial work with success."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.parse
import urllib.request

IDS = {'wJPptuhclZmKAEWs', 'wJPrpFsThWAcBQIw', 'wJQTclrlOLEYYfTB'}

def load(path):
    return json.loads(Path(path).read_text())

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def exact_integer(value, required):
    if type(value) is not int or value != required:
        raise ValueError('Wrong or incomplete integer budget: ' + repr(value))

def full_mapping(record):
    if record.get('complete') is not True:
        raise ValueError('Full mapping training is incomplete')
    for key, number in [('actual_successful_updates', 2250), ('actual_exposures', 18000), ('actual_unique_train', 6000), ('validation_epochs', 3)]:
        exact_integer(record.get(key), number)
    exact_integer(record.get('final_format4_state_tensors'), 467)
    return {'updates': 2250, 'epochs': 3, 'unique_train': 6000, 'state_tensors': 467}

def full_fusion(record):
    if record.get('complete') is not True or record.get('batch6000_complete') is not True:
        raise ValueError('Fusion export is incomplete')
    exact_integer(record.get('included'), 6000)
    exact_integer(record.get('unique_ids'), 6000)
    exact_integer(record.get('unavailable'), 0)
    if record.get('semantic_review_performed') is not False or record.get('human_review_completed') is not False:
        raise ValueError('Source assumptions cannot be relabelled as performed human/semantic review')
    if record.get('source_action_correctness_assumed') is not True:
        raise ValueError('Source-action correctness assumption must be explicitly true')
    return {'included': 6000, 'source_actions_assumed_correct': True}

def pure_video_audit(record):
    if record.get('complete') is not True or record.get('id') not in IDS:
        raise ValueError('Incomplete or unexpected video case')
    media = record['media']
    if media.get('complete_decode') is not True:
        raise ValueError('Full video decode missing')
    for key, number in [('frames', 124), ('height', 1280), ('width', 704), ('audio_streams', 0), ('video_streams', 1)]:
        exact_integer(media.get(key), number)
    exact_number(media.get('fps'), 24)
    guard = record['audio_guard']
    for value in guard['calls'].values():
        exact_integer(value, 0)
    if guard.get('audio_vae_is_none') is not True or guard.get('scheduler_audio_is_none') is not True:
        raise ValueError('Pure-video audio exclusion failed')
    exact_integer(record['sampling'].get('video_scheduler_updates'), 50)
    return record['id']


def exact_number(value, required):
    if type(value) not in (int, float) or value != required:
        raise ValueError('Wrong or boolean numeric configuration: ' + repr(value))

def exact_sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Missing or invalid explicit source SHA')
    return value

def successful_exit(record):
    if record.get('complete') is not True:
        raise ValueError('Completion marker absent or false')
    exact_integer(record.get('exit_code'), 0)

def h3_state_marker(marker, anchor):
    if marker.get('scope') != 'source_label_train':
        raise ValueError('Wrong H3 training state scope')
    exact_integer(marker['progress'].get('optimizer_updates'), 100)
    signature = marker['signature']
    if marker['progress'].get('signature') != signature:
        raise ValueError('H3 progress and configuration signature differ')
    exact_number(signature.get('anchor_lambda'), anchor)
    for key, number in [('world_size', 8), ('seed', 42), ('lora_rank', 32), ('lora_alpha', 64)]:
        exact_integer(signature.get(key), number)
    exact_number(signature.get('learning_rate'), 1e-5)

def anchor_state_receipt(receipt):
    if receipt.get('complete') is not True:
        raise ValueError('Anchor full state verification pending')
    exact_integer(receipt.get('successful_updates'), 100)
    exact_integer(receipt.get('rng_rank_count'), 8)
    for key in ('optimizer_shards', 'model_shards'):
        names = receipt.get(key)
        if (not isinstance(names, list) or len(names) != 8 or len(set(names)) != 8
                or any(not isinstance(name, str) or Path(name).name != name for name in names)):
            raise ValueError('Incomplete or unsafe eight-rank anchor shards: ' + key)
    exact_sha(receipt.get('marker_sha256'))

def anchor_lora_identity(root, entry):
    receipt = load(root / entry['training_receipt_path'])
    anchor_state_receipt(receipt)
    state = root / entry['training_state_path']
    lora = root / entry['lora_path']
    if (Path(receipt['state']).resolve() != state.resolve() or lora.name != 'step-100.safetensors'
            or str(lora) not in receipt.get('lora_files', [])):
        raise ValueError('Anchor receipt is not bound to the explicit training output')
    if file_sha(state / 'complete.json') != receipt['marker_sha256']:
        raise ValueError('Anchor completed state marker SHA differs')
    marker = load(state / 'complete.json')
    h3_state_marker(marker, 0.1)
    files = marker['files']
    required = receipt['optimizer_shards'] + receipt['model_shards'] + ['scheduler.bin', 'custom_checkpoint_0.pkl']
    required += ['random_states_%d.pkl' % rank for rank in range(8)]
    for name in required:
        if name not in files or not (state / name).is_file() or (state / name).stat().st_size != files[name]['bytes']:
            raise ValueError('Anchor completed state file absent or size differs: ' + name)
    terminal = load((root / entry['training_receipt_path']).parent / 'finished.json')
    if terminal.get('complete') is not True:
        raise ValueError('Anchor training queue not complete')
    checksum = exact_sha(file_sha(lora))
    if entry.get('lora_sha256') is not None and entry['lora_sha256'] != checksum:
        raise ValueError('Explicit anchor LoRA SHA differs from the verified actual output')
    return checksum

def caption_identity(entry, caption, raw, raw_sha, upstream=None):
    role = entry['role']
    exact_sha(entry.get('caption_sha256'))
    if exact_sha(entry.get('raw_prediction_sha256')) != raw_sha:
        raise ValueError('Raw music-caption prediction SHA differs')
    if caption.get('raw_captions_unchanged') is not True or caption.get('source_prediction_sha256') != raw_sha:
        raise ValueError('Caption input does not preserve the actual raw predictions')
    if Path(caption.get('source_prediction', '')).resolve() != Path(entry['raw_prediction_path']).resolve():
        raise ValueError('Caption input bound to a different upstream prediction')
    checkpoint = raw.get('source_checkpoint') if role == 'omni_u3000' else raw.get('checkpoint')
    expected = 'checkpoint-u0003000-e04-p0000000' if role == 'omni_u3000' else 'checkpoint-u0003750-e05-p0000000'
    if not isinstance(checkpoint, str) or Path(checkpoint).name != expected:
        raise ValueError('Omni role and actual upstream checkpoint differ')
    if role == 'omni_u3000':
        exact_integer(raw.get('updates'), 3000)
        if raw.get('human_review') is not False:
            raise ValueError('Unexpected claimed Omni human review')
        if (exact_sha(raw.get('checkpoint_adapter_sha256')) != exact_sha(entry.get('checkpoint_adapter_sha256'))
                or exact_sha(raw.get('source_checkpoint_marker_sha256')) != exact_sha(entry.get('checkpoint_marker_sha256'))):
            raise ValueError('Omni u3000 deployed model source differs')
    else:
        if upstream is None:
            raise ValueError('Final u3750 upstream completion evidence absent')
        successful_exit(upstream)
        if upstream.get('no_GT_in_user_input') is not True or upstream.get('actual_training_added') is not False:
            raise ValueError('Final upstream inference used wrong input/training protocol')
    rows, predictions = caption.get('records'), raw.get('predictions')
    if not isinstance(rows, list) or not isinstance(predictions, list) or len(rows) != 3 or len(predictions) != 3:
        raise ValueError('Require three actual raw music-caption cases')
    if {row['id'] for row in rows} != IDS or {row['id'] for row in predictions} != IDS:
        raise ValueError('Omni fixed three source IDs differ')
    lookup = {row['id']: row for row in predictions}
    for row in rows:
        if row.get('motion_caption') != lookup[row['id']].get('caption'):
            raise ValueError('H3 caption was edited or swapped from the actual music prediction')


def pure_model_sources(entry, report, expected_lora=None):
    if report.get('complete') is not True or report.get('protocol') != 'h3_caption_image_infer_v1':
        raise ValueError('Pure-video inference run incomplete or wrong protocol')
    if set(report.get('completed_ids', [])) != IDS or len(report.get('completed_ids', [])) != 3:
        raise ValueError('Pure-video inference run lacks all fixed three IDs')
    caption_source = report.get('caption_source', {})
    if caption_source.get('sha256') != exact_sha(entry.get('caption_sha256')):
        raise ValueError('Evaluated H3 caption source differs from explicit role source')
    if Path(caption_source.get('path', '')).resolve() != Path(entry['caption_path']).resolve():
        raise ValueError('Evaluated H3 caption source path differs')
    if entry['role'] in ('omni_u3000', 'omni_final3750'):
        if report.get('lora_source') is not None:
            raise ValueError('Frozen original H3 Omni comparisons cannot load a LoRA')
    else:
        checksum = exact_sha(expected_lora if expected_lora is not None else entry.get('lora_sha256'))
        if (report.get('lora_source') or {}).get('file', {}).get('sha256') != checksum:
            raise ValueError('Evaluated H3 LoRA does not match the verified actual training output')


def native_state_receipt(receipt, checkpoint_sha):
    for key in ('complete', 'actual_three_unique_epochs', 'full_saved_Adam_RNG_sampler_pointer', 'strict_loaded_complete_MusicFM'):
        if receipt.get(key) is not True:
            raise ValueError('Incomplete native final state receipt: ' + key)
    for key, number in [('steps', 2250), ('exposures', 18000), ('optimizer_states', 402), ('state_tensors', 467), ('trainable_tensors', 402)]:
        exact_integer(receipt.get(key), number)
    checksum = exact_sha(checkpoint_sha)
    if receipt.get('checkpoint_sha256') != checksum or receipt.get('local_stage_actual_sha256') != checksum:
        raise ValueError('Native model, full state receipt and deployed local stage differ')
    metadata = receipt.get('metadata', {})
    expected = dict(format_version=4, state_scope='direct_music_to_h3_text', backend='musicfm',
                    context_policy='forbidden', freeze_backbone=False, adapter='linear_1024_5120_bias',
                    output_normalization='none_raw_hidden')
    for key, value in expected.items():
        if type(metadata.get(key)) is not type(value) or metadata[key] != value:
            raise ValueError('Native final metadata scope differs: ' + key)
    if metadata.get('online_training', {}).get('train_backbone') is not True:
        raise ValueError('Native full dense train-backbone receipt missing')


def native_case_audit(record, raw, checkpoint_sha, expected_case=None):
    if record.get('sample_id') not in IDS:
        raise ValueError('Unexpected native source ID')
    for key in ('full_media_decode_checked', 'full_video_audio_decode'):
        if record.get(key) is not True:
            raise ValueError('Native full video/audio decode missing: ' + key)
    exact_integer(record.get('ffmpeg_decoded_video_frames'), 124)
    streams = record.get('streams', [])
    video = [stream for stream in streams if stream.get('codec_type') == 'video']
    if len(video) != 1 or not any(stream.get('codec_type') == 'audio' for stream in streams):
        raise ValueError('Original native AV requires one video plus actual audio')
    for key, number in [('width', 704), ('height', 1280), ('decoded_frames', 124)]:
        exact_integer(video[0].get(key), number)
    exact_number(video[0].get('fps'), 24)
    if raw.get('complete') is not True:
        raise ValueError('Native raw audit incomplete')
    inputs, audit_record = raw['input'], raw['audit']
    if expected_case is not None and inputs.get('case') != expected_case:
        raise ValueError('Native original image/audio/prefix case differs from the fixed source manifest')
    if inputs.get('checkpoint_sha256') != exact_sha(checkpoint_sha) or inputs['case'].get('sample_id') != record['sample_id']:
        raise ValueError('Native raw audit actual checkpoint/case differs')
    if record.get('sha256') != raw.get('video_sha256'):
        raise ValueError('Native validated and raw media SHA differ')
    for key in ('student_reads_target_or_caption', 'teacher_identity', 'student_zero_input'):
        if inputs.get(key) is not False:
            raise ValueError('Native student target/caption/zero-input protocol violated: ' + key)
    if inputs.get('output_audio') != 'original_mux' or raw.get('mux_audio_sha256') != inputs.get('native_waveform_sha256'):
        raise ValueError('Original native reference waveform and mux differ')
    if audit_record.get('native_audio_condition') != 'real_reference_audio' or audit_record.get('music_mode') != 'ref':
        raise ValueError('Original native real reference audio changed')
    for key in ('full_cached_prefix_preserved', 'original_cached_prefix_tags_preserved', 'packed_text_tags_exact',
                'native_audio_vae_unchanged', 'original_joint_pipeline'):
        if audit_record.get(key) is not True:
            raise ValueError('Native pipeline protocol changed: ' + key)
    exact_integer(audit_record.get('text_encoder_calls'), 0)
    if audit_record.get('semantic_text') is not None:
        raise ValueError('Native direct inference cannot restore semantic text')
    for key in ('video', 'audio'):
        exact_integer(audit_record['scheduler_updates'].get(key), 50)


def video_configurations(config):
    roles = {'purefm100', 'anchor01_100', 'omni_u3000', 'direct_full2250', 'omni_final3750'}
    if not isinstance(config, list) or len(config) != 5 or {row.get('role') for row in config} != roles:
        raise ValueError('Require five explicit distinct model roles')
    if len({row.get('model_id') for row in config}) != 5:
        raise ValueError('Duplicate model configuration')
    for row in config:
        if row.get('role') == 'direct_full2250':
            exact_sha(row.get('case_manifest_sha256'))
            if not isinstance(row.get('case_manifest_path'), str) or not row['case_manifest_path']:
                raise ValueError('Missing explicit original native fixed-three source manifest')
        kind = 'native_AV' if row['role'] == 'direct_full2250' else 'pure_video'
        if row.get('kind') != kind or not isinstance(row.get('directory'), str):
            raise ValueError('Wrong role protocol or missing explicit directory')
        path = Path(row['directory'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Configuration directory must stay within the explicit experiment root')
        expected_id = 'MS/' + row['directory'] + ('/videos' if kind == 'pure_video' else '')
        if row.get('model_id') != expected_id:
            raise ValueError('Model registration ID and actual directory differ')
        if kind == 'pure_video':
            exact_sha(row.get('caption_sha256'))
            if not isinstance(row.get('caption_path'), str) or not row['caption_path']:
                raise ValueError('Missing explicit caption input path')
        if row['role'] == 'purefm100':
            exact_sha(row.get('lora_sha256'))
        if row['role'] == 'anchor01_100':
            if any(not isinstance(row.get(k), str) or not row[k] for k in ('training_receipt_path', 'training_state_path', 'lora_path')):
                raise ValueError('Missing explicit actual anchor training evidence paths')
        if row['role'] in ('omni_u3000', 'omni_final3750'):
            exact_sha(row.get('raw_prediction_sha256'))
            if not isinstance(row.get('raw_prediction_path'), str) or not row['raw_prediction_path']:
                raise ValueError('Missing actual raw prediction evidence path')
        if row['role'] == 'omni_u3000':
            exact_sha(row.get('checkpoint_adapter_sha256')); exact_sha(row.get('checkpoint_marker_sha256'))
        if row['role'] == 'omni_final3750' and not isinstance(row.get('upstream_finished_path'), str):
            raise ValueError('Missing final u3750 upstream completion path')
    return {'roles': sorted(roles), 'models': 5, 'fixed_cases_per_model': 3, 'total_cases': 15}

def self_test():
    import copy
    import tempfile
    checks = []
    def reject(name, operation):
        try:
            operation()
        except (ValueError, KeyError, TypeError, OSError):
            checks.append(name)
        else:
            raise AssertionError('Invalid completion accepted: ' + name)
    good = dict(complete=True, actual_successful_updates=2250, actual_exposures=18000,
                actual_unique_train=6000, validation_epochs=3, final_format4_state_tensors=467)
    full_mapping(good)
    for key, value in [('complete', 1), ('actual_successful_updates', 100), ('actual_exposures', True), ('validation_epochs', 2)]:
        reject('mapping_' + key, lambda k=key, v=value: full_mapping({**good, k: v}))
    successful_exit(dict(complete=True, exit_code=0))
    for value in (False, True, '0', 1):
        reject('exit_type_' + repr(value), lambda v=value: successful_exit(dict(complete=True, exit_code=v)))
    fusion = dict(complete=True, batch6000_complete=True, included=6000, unique_ids=6000, unavailable=0,
                  semantic_review_performed=False, human_review_completed=False, source_action_correctness_assumed=True)
    full_fusion(fusion)
    for key, value in [('unique_ids', 5999), ('human_review_completed', True), ('batch6000_complete', 'true'),
                       ('source_action_correctness_assumed', False), ('source_action_correctness_assumed', 1)]:
        reject('fusion_' + key, lambda k=key, v=value: full_fusion({**fusion, k: v}))
    reject('fusion_missing_assumption', lambda: full_fusion({k: v for k, v in fusion.items() if k != 'source_action_correctness_assumed'}))
    sid = sorted(IDS)[0]
    video = dict(complete=True, id=sid, media=dict(complete_decode=True, frames=124, height=1280, width=704,
        audio_streams=0, video_streams=1, fps=24), audio_guard=dict(calls=dict(audio_patch=0),
        audio_vae_is_none=True, scheduler_audio_is_none=True), sampling=dict(video_scheduler_updates=50))
    pure_video_audit(video)
    for section, key, value in [('media', 'frames', 123), ('media', 'complete_decode', 1), ('media', 'audio_streams', 1),
                                ('sampling', 'video_scheduler_updates', 49), ('audio_guard', 'audio_vae_is_none', 1)]:
        changed = copy.deepcopy(video); changed[section][key] = value
        reject('video_' + key, lambda v=changed: pure_video_audit(v))
    signature = dict(anchor_lambda=0.1, world_size=8, seed=42, lora_rank=32, lora_alpha=64, learning_rate=1e-5)
    marker = dict(scope='source_label_train', signature=signature, progress=dict(optimizer_updates=100, signature=signature))
    h3_state_marker(marker, 0.1)
    for value in (False, True, '0.1'):
        changed = copy.deepcopy(marker); changed['signature']['anchor_lambda'] = value
        reject('anchor_numeric_' + repr(value), lambda v=changed: h3_state_marker(v, 0.1))
    pure_marker = copy.deepcopy(marker); pure_marker['signature']['anchor_lambda'] = False
    reject('purefm_boolean_anchor', lambda: h3_state_marker(pure_marker, 0.0))
    receipt = dict(complete=True, successful_updates=100, rng_rank_count=8, marker_sha256='a' * 64,
        optimizer_shards=['optim_%d.pt' % r for r in range(8)], model_shards=['model_%d.pt' % r for r in range(8)])
    anchor_state_receipt(receipt)
    for key, value in [('rng_rank_count', True), ('rng_rank_count', 1), ('optimizer_shards', []), ('model_shards', ['same'] * 8),
                       ('successful_updates', 99), ('complete', 1)]:
        reject('anchor_receipt_' + key, lambda k=key, v=value: anchor_state_receipt({**receipt, k: v}))
    raw_sha, cap_sha, lora_sha = 'a' * 64, 'b' * 64, 'c' * 64
    caption_entry = dict(role='omni_u3000', caption_path='/fixture/caps.json', caption_sha256=cap_sha,
        raw_prediction_path='/fixture/raw.json', raw_prediction_sha256=raw_sha,
        checkpoint_adapter_sha256='d' * 64, checkpoint_marker_sha256='e' * 64)
    predictions = [dict(id=s, caption='Unedited ' + s) for s in sorted(IDS)]
    caption = dict(raw_captions_unchanged=True, source_prediction='/fixture/raw.json', source_prediction_sha256=raw_sha,
                   records=[dict(id=p['id'], motion_caption=p['caption']) for p in predictions])
    raw = dict(updates=3000, source_checkpoint='/fixture/checkpoint-u0003000-e04-p0000000', human_review=False,
               checkpoint_adapter_sha256='d' * 64, source_checkpoint_marker_sha256='e' * 64, predictions=predictions)
    caption_identity(caption_entry, caption, raw, raw_sha)
    for key in ('caption_sha256', 'raw_prediction_sha256'):
        reject('Omni_missing_' + key, lambda k=key: caption_identity({k2: v for k2, v in caption_entry.items() if k2 != k}, caption, raw, raw_sha))
    final_entry = {**caption_entry, 'role': 'omni_final3750'}
    upstream = dict(complete=True, exit_code=0, no_GT_in_user_input=True, actual_training_added=False)
    final_raw = dict(checkpoint='/fixture/checkpoint-u0003750-e05-p0000000', predictions=predictions)
    caption_identity(final_entry, caption, final_raw, raw_sha, upstream)
    reject('swap_u3000_as_final3750', lambda: caption_identity(final_entry, caption, raw, raw_sha, upstream))
    reject('swap_final3750_as_u3000', lambda: caption_identity(caption_entry, caption, final_raw, raw_sha))
    reject('final_upstream_exit_false', lambda: caption_identity(final_entry, caption, final_raw, raw_sha, {**upstream, 'exit_code': False}))
    reject('final_upstream_GT_bool1', lambda: caption_identity(final_entry, caption, final_raw, raw_sha, {**upstream, 'no_GT_in_user_input': 1}))
    edited = copy.deepcopy(caption); edited['records'][0]['motion_caption'] = 'Edited'
    reject('caption_edited_body', lambda: caption_identity(caption_entry, edited, raw, raw_sha))
    report = dict(complete=True, protocol='h3_caption_image_infer_v1', completed_ids=sorted(IDS),
        caption_source=dict(path='/fixture/caps.json', sha256=cap_sha), lora_source=None)
    pure_model_sources(caption_entry, report)
    reject('frozen_Omni_with_LoRA', lambda: pure_model_sources(caption_entry, {**report, 'lora_source': {'file': {'sha256': lora_sha}}}))
    anchor_entry = {**caption_entry, 'role': 'anchor01_100'}
    anchor_report = {**report, 'lora_source': {'file': {'sha256': lora_sha}}}
    pure_model_sources(anchor_entry, anchor_report, lora_sha)
    reject('wrong_anchor_lora', lambda: pure_model_sources(anchor_entry, anchor_report, 'f' * 64))
    reject('missing_anchor_actual_lora', lambda: pure_model_sources(anchor_entry, anchor_report))
    reject('wrong_caption_file', lambda: pure_model_sources(caption_entry, {**report, 'caption_source': {'path': '/fixture/caps.json', 'sha256': 'f' * 64}}))
    metadata = dict(format_version=4, state_scope='direct_music_to_h3_text', backend='musicfm', context_policy='forbidden',
                    freeze_backbone=False, adapter='linear_1024_5120_bias', output_normalization='none_raw_hidden',
                    online_training=dict(train_backbone=True))
    native = dict(complete=True, actual_three_unique_epochs=True, full_saved_Adam_RNG_sampler_pointer=True,
        strict_loaded_complete_MusicFM=True, steps=2250, exposures=18000, optimizer_states=402, state_tensors=467,
        trainable_tensors=402, checkpoint_sha256=raw_sha, local_stage_actual_sha256=raw_sha, metadata=metadata)
    native_state_receipt(native, raw_sha)
    for key, value in [('steps', 2249), ('state_tensors', 1), ('exposures', 1), ('optimizer_states', True),
                       ('actual_three_unique_epochs', False), ('full_saved_Adam_RNG_sampler_pointer', 1), ('trainable_tensors', 401)]:
        reject('native_state_' + key, lambda k=key, v=value: native_state_receipt({**native, k: v}, raw_sha))
    changed = copy.deepcopy(native); changed['metadata']['freeze_backbone'] = 0
    reject('native_metadata_false_as_zero', lambda: native_state_receipt(changed, raw_sha))
    checked = dict(sample_id=sid, full_media_decode_checked=True, full_video_audio_decode=True, ffmpeg_decoded_video_frames=124,
        sha256=cap_sha, streams=[dict(codec_type='video', width=704, height=1280, decoded_frames=124, fps=24), dict(codec_type='audio')])
    audit_record = dict(scheduler_updates=dict(video=50, audio=50), native_audio_condition='real_reference_audio', music_mode='ref',
        text_encoder_calls=0, semantic_text=None, **{k: True for k in ('full_cached_prefix_preserved', 'original_cached_prefix_tags_preserved',
        'packed_text_tags_exact', 'native_audio_vae_unchanged', 'original_joint_pipeline')})
    native_raw = dict(complete=True, video_sha256=cap_sha, mux_audio_sha256='wave',
        input=dict(checkpoint_sha256=raw_sha, case=dict(sample_id=sid), student_reads_target_or_caption=False,
        teacher_identity=False, student_zero_input=False, output_audio='original_mux', native_waveform_sha256='wave'), audit=audit_record)
    native_case_audit(checked, native_raw, raw_sha, dict(sample_id=sid))
    reject('native_original_song_swap', lambda: native_case_audit(checked, native_raw, raw_sha, dict(sample_id=sid, audio_path='different-song')))
    for section, key, value in [('input', 'student_reads_target_or_caption', True), ('input', 'teacher_identity', 0),
        ('input', 'student_zero_input', 0), ('input', 'output_audio', 'generated'), ('audit', 'native_audio_condition', 'silence'),
        ('audit', 'original_joint_pipeline', 1), ('audit', 'text_encoder_calls', False)]:
        changed = copy.deepcopy(native_raw); changed[section][key] = value
        reject('native_case_' + key, lambda v=changed: native_case_audit(checked, v, raw_sha))
    reject('native_decoded_bool1', lambda: native_case_audit({**checked, 'full_video_audio_decode': 1}, native_raw, raw_sha))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); state = root / 'state'; state.mkdir(); lora = root / 'step-100.safetensors'; lora.write_bytes(b'CPU fixture')
        names = receipt['optimizer_shards'] + receipt['model_shards'] + ['scheduler.bin', 'custom_checkpoint_0.pkl']
        names += ['random_states_%d.pkl' % r for r in range(8)]
        for name in names: (state / name).write_bytes(b'fixture')
        state_marker = {**marker, 'files': {name: {'bytes': 7} for name in names}}
        (state / 'complete.json').write_text(json.dumps(state_marker))
        actual_receipt = {**receipt, 'state': str(state), 'lora_files': [str(lora)], 'marker_sha256': file_sha(state / 'complete.json')}
        (root / 'verified.json').write_text(json.dumps(actual_receipt)); (root / 'finished.json').write_text(json.dumps({'complete': True}))
        explicit = dict(training_receipt_path='verified.json', training_state_path='state', lora_path='step-100.safetensors')
        assert anchor_lora_identity(root, explicit) == file_sha(lora)
        reject('anchor_explicit_wrong_sha', lambda: anchor_lora_identity(root, {**explicit, 'lora_sha256': 'f' * 64}))
        (state / names[0]).unlink()
        reject('anchor_state_missing_actual_file', lambda: anchor_lora_identity(root, explicit))
    fixture_config = []
    for role in ('purefm100', 'anchor01_100', 'omni_u3000', 'direct_full2250', 'omni_final3750'):
        kind = 'native_AV' if role == 'direct_full2250' else 'pure_video'
        row = dict(role=role, kind=kind, directory=role, model_id='MS/' + role + ('/videos' if kind == 'pure_video' else ''),
            caption_path='/fixture/caps.json', caption_sha256=cap_sha, lora_sha256=lora_sha,
            training_receipt_path='verified.json', training_state_path='state', lora_path='step-100.safetensors',
            raw_prediction_path='/fixture/raw.json', raw_prediction_sha256=raw_sha,
            checkpoint_adapter_sha256='d' * 64, checkpoint_marker_sha256='e' * 64,
            upstream_finished_path='/fixture/finished.json', case_manifest_path='/fixture/joint.json', case_manifest_sha256=raw_sha)
        fixture_config.append(row)
    video_configurations(fixture_config)
    for index, key in ((0, 'lora_sha256'), (1, 'training_receipt_path'), (2, 'raw_prediction_sha256'), (3, 'case_manifest_sha256'), (4, 'caption_sha256')):
        changed = copy.deepcopy(fixture_config); del changed[index][key]
        reject('roles_missing_' + key, lambda v=changed: video_configurations(v))
    changed = copy.deepcopy(fixture_config); changed[3]['kind'] = 'pure_video'
    reject('roles_native_as_pure_video', lambda: video_configurations(changed))
    changed = copy.deepcopy(fixture_config); changed[4]['model_id'] = changed[2]['model_id']
    reject('roles_duplicate_registered_model', lambda: video_configurations(changed))
    print(json.dumps(dict(self_test='passed', fixture_only=True, negative_cases=len(checks), rejected=checks,
        actual_GPU_or_private_model_Adam_loaded=False, source_sha256=file_sha(__file__))), flush=True)

def audit(args):
    root = args.ms
    checks = []
    def run(name, operation):
        try:
            result = operation()
            checks.append({'name': name, 'complete': True, 'evidence': result})
        except (OSError, ValueError, KeyError, TypeError, AssertionError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            checks.append({'name': name, 'complete': False, 'error': str(error)})
    def n7():
        record = load(root / 'omni_ms_style_full_queue_v4/finished.json')
        successful_exit(record)
        state = record['training_complete']
        exact_integer(state.get('updates'), 3750); exact_integer(state.get('full_epochs'), 5)
        return {'updates': state['updates'], 'full_epochs': state['full_epochs'], 'stop_reason': state['stop_reason'], 'quality_accepted': state.get('human_quality_accepted')}
    run('N7_original_MS_full_training', n7)
    run('N2_full_training_budget_terminal', lambda: full_mapping(load(root / 'music_direct_token_full6k_n2_v1/finished.json')))
    def mapping():
        directory = root / 'music_direct_token_full6k_n2_v1'
        record = load(directory / 'finished.json'); result = full_mapping(record)
        latest = load(directory / 'latest.json'); exact_integer(latest.get('steps'), 2250)
        receipt = load(root / 'music_direct_full6k_final2250_video_n4_queue_v1/final_cpu_checkpoint.json')
        native_state_receipt(receipt, latest['checkpoint_sha256'])
        if latest.get('checkpoint_name') != 'model_step02250.pt' or latest.get('train_state_name') != 'train_state_step02250.pt':
            raise ValueError('Only the immutable actual final2250 model/state pair is accepted')
        model = directory / latest['checkpoint_name']
        if record.get('final_checkpoint') != str(model):
            raise ValueError('Training finished and final model pointer differ')
        if latest['checkpoint_sha256'] != file_sha(model): raise ValueError('Final model SHA differs')
        if not (directory / latest['train_state_name']).is_file(): raise ValueError('Final optimizer/RNG state absent')
        for epoch in range(1, 4):
            validation = load(directory / ('validation_epoch%02d.json' % epoch))
            exact_integer(validation.get('validation_samples'), 500)
            exact_integer(validation.get('epoch'), epoch)
            exact_integer(validation.get('optimizer_updates'), epoch * 750)
        return {**result, 'final_model': str(model), 'final_model_SHA': latest['checkpoint_sha256']}
    run('N2_full_mapping_three_epochs', mapping)
    def fusion():
        directory = args.fusion_export
        record = load(directory / 'complete.json'); result = full_fusion(record)
        for name, expected in record['files'].items():
            if file_sha(directory / name) != expected: raise ValueError('Export file SHA differs: ' + name)
        rows = [json.loads(line) for line in (directory / 'captions.jsonl').read_text().splitlines() if line.strip()]
        exact_integer(len(rows), 6000); exact_integer(len({row['id'] for row in rows}), 6000)
        if any(row.get('split') != 'train' for row in rows): raise ValueError('Fusion split changed')
        return result
    run('N1_complete_fusion_export', fusion)
    def sidecar():
        directory = root / 'dance_caption_fusion6k_dimensions_v1'
        rows = [json.loads(line) for line in (directory / 'context_rows.jsonl').read_text().splitlines() if line.strip()]
        exact_integer(len(rows), 6000); exact_integer(len({row['id'] for row in rows}), 6000)
        return {'metadata_only_rows': 6000, 'current_caption_injection_claimed': False}
    run('music_motion_metadata_sidecar', sidecar)
    def h3_training():
        pure = root / args.purefm_training
        marker = load(pure / 'training-state-final-100/complete.json')
        h3_state_marker(marker, 0.0)
        if not (pure / 'step-100.safetensors').is_file():
            raise ValueError('PureFM deployment LoRA absent')
        config = load(args.video_config)
        anchor_entry = next(entry for entry in config if entry['role'] == 'anchor01_100')
        if (root / anchor_entry['training_receipt_path']).parent.resolve() != (root / args.anchor_queue).resolve():
            raise ValueError('Explicit anchor evidence differs from the selected actual queue')
        checksum = anchor_lora_identity(root, anchor_entry)
        return {'pureFM_updates': 100, 'anchor_updates': 100, 'actual_anchor_lora_sha256': checksum}
    run('H3_pureFM_and_anchor_training', h3_training)
    video_config = load(args.video_config)
    run('five_video_configurations', lambda: video_configurations(video_config))
    for entry in video_config:
        def videos(entry=entry):
            directory = root / entry['directory']
            if entry['role'] == 'direct_full2250' and entry['kind'] != 'native_AV':
                raise ValueError('Direct mapping must retain original native AV protocol')
            if entry['role'] != 'direct_full2250' and entry['kind'] != 'pure_video':
                raise ValueError('Caption H3 comparisons must retain pure-video protocol')
            if entry['kind'] == 'pure_video':
                terminal = load(directory / 'finished.json')
                if terminal.get('complete') is not True: raise ValueError('Video job not complete')
                records = [load(directory / 'videos' / (sid + '.audit.json')) for sid in IDS]
                ids = {pure_video_audit(record) for record in records}
                inference_report = load(directory / 'videos/run_complete.json')
                caption_path = Path(entry['caption_path'])
                if file_sha(caption_path) != exact_sha(entry.get('caption_sha256')):
                    raise ValueError('Actual caption file SHA differs from explicit role identity')
                expected_lora = None
                if entry['role'] == 'purefm100':
                    expected_lora = file_sha(root / args.purefm_training / 'step-100.safetensors')
                    if expected_lora != exact_sha(entry.get('lora_sha256')):
                        raise ValueError('PureFM expected LoRA does not match actual training output')
                elif entry['role'] == 'anchor01_100':
                    expected_lora = anchor_lora_identity(root, entry)
                else:
                    raw_path = Path(entry['raw_prediction_path'])
                    raw_sha = file_sha(raw_path)
                    upstream = load(entry['upstream_finished_path']) if entry['role'] == 'omni_final3750' else None
                    caption_identity(entry, load(caption_path), load(raw_path), raw_sha, upstream)
                pure_model_sources(entry, inference_report, expected_lora)
                if ids != IDS: raise ValueError('Expected all fixed three cases')
                for record in records:
                    if file_sha(record['video']['path']) != record['video']['sha256']: raise ValueError('Published video SHA differs')
            else:
                terminal = load(directory / 'report.json')
                if terminal.get('complete') is not True or terminal.get('full_media_decode_checked') is not True: raise ValueError('Native AV decode report incomplete')
                exact_integer(terminal.get('actual_final_updates'), 2250)
                checkpoint_receipt = load(root / 'music_direct_full6k_final2250_video_n4_queue_v1/final_cpu_checkpoint.json')
                native_state_receipt(checkpoint_receipt, terminal['checkpoint_sha256'])
                case_path = Path(entry['case_manifest_path'])
                if file_sha(case_path) != exact_sha(entry.get('case_manifest_sha256')):
                    raise ValueError('Original native fixed-three source manifest SHA differs')
                case_manifest = load(case_path)
                expected_cases = {case['sample_id']: case for case in case_manifest['cases']}
                if case_manifest.get('condition') != 'joint' or set(expected_cases) != IDS or len(case_manifest['cases']) != 3:
                    raise ValueError('Original native joint fixed-three source manifest changed')
                records = terminal['cases']; ids = {record['sample_id'] for record in records}
                for record in records:
                    raw_audit = load(record['audit_path'])
                    native_case_audit(record, raw_audit, terminal['checkpoint_sha256'], expected_cases[record['sample_id']])
                if ids != IDS or len(records) != 3: raise ValueError('Expected all fixed three native cases')
                for record in records:
                    if record.get('full_media_decode_checked') is not True or file_sha(record['path']) != record['sha256']: raise ValueError('Native video verification missing')
            return {'model_id': entry['model_id'], 'cases': sorted(ids), 'kind': entry['kind']}
        run('videos:' + entry['model_id'], videos)
    def website():
        if args.offline:
            raise ValueError('Review HTTP requires connected final verification; offline check is pending')
        with urllib.request.urlopen(args.review_base.rstrip('/') + '/manifest', timeout=15) as response:
            data = response.read(); status = response.status
            if response.headers.get('Content-Encoding') == 'gzip': data = gzip.decompress(data)
        manifest = json.loads(data); models = {row['id']: row for row in manifest['models']}
        if status != 200: raise ValueError('Manifest not accessible')
        total = 0
        for entry in video_config:
            model = models[entry['model_id']]
            if set(model['conds']['joint']) != IDS: raise ValueError('Review model lacks three cases')
            for sid in IDS:
                filename = model['conds']['joint'][sid][0]
                path = root.parent / model['id'] / 'joint' / filename
                url = args.review_base.rstrip('/') + '/media?' + urllib.parse.urlencode({'p': str(path)})
                request = urllib.request.Request(url, headers={'Range': 'bytes=0-1023'})
                with urllib.request.urlopen(request, timeout=15) as response:
                    body = response.read(); code = response.status
                if code != 206 or len(body) != 1024: raise ValueError('Review video stream invalid')
                total += 1
        return {'manifest_HTTP': 200, 'verified_video_streams': total, 'all_model_cases_registered': True}
    run('review_HTTP_all_five_configurations', website)
    def git():
        if args.offline:
            raise ValueError('GitHub HEAD requires connected main-session verification; offline check is pending')
        local = subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip()
        remote = subprocess.check_output(['git', '-C', str(args.repo), 'ls-remote', 'origin', 'refs/heads/main'], text=True, timeout=30).split()[0]
        if local != remote: raise ValueError('GitHub HEAD differs from local verified commit')
        return {'HEAD': local, 'origin_main': remote}
    run('GitHub_HEAD_matches', git)
    result = {'checker_revision': 2, 'complete': bool(checks) and all(check['complete'] is True for check in checks), 'checks': checks,
              'human_quality_is_separate_from_technical_completion': True, 'pending': [c['name'] for c in checks if not c['complete']]}
    with args.out.open('x') as stream: json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result['complete']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--offline', action='store_true', help='Do not contact review/GitHub; leave those checks pending')
    parser.add_argument('--ms', type=Path)
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--fusion-export', type=Path)
    parser.add_argument('--purefm-training')
    parser.add_argument('--anchor-queue')
    parser.add_argument('--video-config', type=Path)
    parser.add_argument('--review-base')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.self_test: self_test(); return
    required = ['ms','repo','fusion_export','purefm_training','anchor_queue','video_config','review_base','out']
    if any(getattr(args,name) is None for name in required): parser.error('All completion paths and review URL must be explicitly supplied')
    raise SystemExit(0 if audit(args) else 2)

if __name__ == '__main__': main()
