"""Publish independently verified eval cases without replacing human review fields."""
import argparse
import copy
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
EXP = ROOT / 'diff/logs_omnidance/exp'
MANIFEST = EXP / 'review_web/manifest_h3_ms_style_gt_eval3_v1.json'
IDS = ['wJPptuhclZmKAEWs', 'wJPrpFsThWAcBQIw', 'wJQTclrlOLEYYfTB']
GT = 'MS/h3_ms_style_eval3_gt_videos_v1'
GT_CAPTION = 'MS/omni_h3_ms_style_eval3_v1/h3_gt_caption3.json'
PURE_SHA = 'e5aef58b06bba1b0f8b570fb61cfe053a85d48f4d734c32e7cd923c6931749f0'
DIRECT_SHA = '127a570f1422ee2f28f78588ace1c886da7ebc68e76ae421672148c4ab0f6cf6'
JOBS = [
    dict(id='MS/review_h3_purefm100_n4_chunked_v2/videos', kind='purefm',
         label='H3·原MS＋图像·pureFM@100', caption=GT_CAPTION,
         user_feedback='h3_purefm100_and_omni_u3000_user_review_20261024.json',
         desc='原MS＋参考图，H3 pureFM实际100更新；50步/seed42/CFG1/704×1280/124帧@24fps。内部无音频。用户已反馈有效舞蹈且幅度更大；定性反馈不等于数字评分。',
         preset='【H3动作学习@100】pureFM 与原MS冻结H3纯视频基线', baselines=[GT]),
    dict(id='MS/review_omni6k_u3000_n6_video_v2/videos', kind='omni',
         label='Omni30B·6k u3000→冻结H3纯视频',
         user_feedback='h3_purefm100_and_omni_u3000_user_review_20261024.json',
         caption='MS/review_omni6k_u3000_n6_video_v2/captions.json',
         prediction='MS/n7_omni6k_u3000_eval_n6_v1/omni_prediction3_raw.json',
         checkpoint='checkpoint-u0003000-e04-p0000000',
         desc='真实音乐→Omni u3000原始caption→同图冻结H3纯视频；中间权重，不是最终u3750。原始重复正文未改；内部无音频。用户已反馈动作重复偏弱；不填数字评分。',
         preset='【音乐到动作caption·6k u3000】预测caption与原MS纯视频基线', baselines=[GT]),
    dict(id='MS/review_omni6k_final_u3750_n6_v1/videos', kind='omni',
         label='Omni30B·6k 最终u3750→冻结H3纯视频',
         user_feedback='omni_final3750_user_review_20261024.json',
         caption='MS/review_omni6k_final_u3750_n6_v1/captions.json',
         prediction='MS/omni_ms_style_full_eval3_queue_v2/full_predictions.json',
         checkpoint='checkpoint-u0003750-e05-p0000000',
         desc='Omni完整6k最终u3750真实音乐预测原始caption→同图冻结H3；不是u3000。正文未改；50步/seed42/CFG1/704×1280/124帧@24fps，内部无音频。用户反馈三个视频有部分动作重复，怀疑caption重复；不是全部动作失败，原因未经受控消融证实，未给最终对u3000排名或数字评分。',
         preset='【音乐到动作caption·最终u3750】最终、u3000与原MS纯视频基线',
         baselines=['MS/review_omni6k_u3000_n6_video_v2/videos', GT]),
    dict(id='MS/review_h3_anchor01_s100_n4_v3/videos', kind='anchor',
         label='H3·原MS＋图像·anchor0.1@100', caption=GT_CAPTION,
         training_report='MS/h3_anchor100_independent_audit_v4_20261024/full_state_verified.json',
         training_report_sha256='9609fb8738a5f28e91e7b9916438b1f77fdfa40f4920ac58b30aedbce0594d58',
         desc='anchor0.1真实100更新、完整训练state来源核验后的原MS＋参考图H3纯视频；actual v3，失败v2不升格。50步/seed42/CFG1/704×1280/124帧@24fps，内部无音频；人工评审待定。',
         preset='【H3动作学习@100】anchor0.1、pureFM与原MS冻结H3纯视频基线',
         baselines=['MS/review_h3_purefm100_n4_chunked_v2/videos', GT]),
    dict(id='MS/music_direct_full6k_final2250_video_n4_v3', kind='native',
         label='音乐direct token·完整6k三轮@2250',
         receipt='MS/music_direct_full6k_final2250_video_n4_queue_v3/final_cpu_checkpoint.json',
         desc='N2实际2250更新/18000曝光/6k三轮完整Adam/RNG/MusicFM来源核验，N4 actual v3固定三case。原生real reference audio AVjoint，完整cachedP/tags、学生不读GT/caption、TC调用0、video/audio各50步，original_mux真实配乐；非纯视频协议。人工视频评审待定。',
         preset='【音乐direct token·完整6k三轮@2250】最终模型与旧mean/token@100',
         baselines=['MS/music_direct_video_compare_n4_v1/mean', 'MS/music_direct_video_compare_n4_v1/token']),
]


class Pending(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise Pending(message)


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def record(path):
    value = read(path)
    need(isinstance(value, dict), 'missing/incomplete JSON: ' + str(path))
    return value


_HASHES = {}


def sha(path):
    path = Path(path)
    stat = path.stat()
    key = (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if key not in _HASHES:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        current = path.stat()
        need((current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns) == (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns), 'file changed while hashing: ' + str(path))
        _HASHES[key] = h.hexdigest()
    return _HASHES[key]


def bound_file(path, digest):
    need(isinstance(digest, str) and len(digest) == 64, 'missing SHA: ' + str(path))
    need(Path(path).is_file() and sha(path) == digest, 'file SHA mismatch: ' + str(path))


def source_file(source, expected=None):
    need(isinstance(source, dict), 'missing source file record')
    if expected is not None:
        need(source['path'] == str(expected), 'source path swapped')
    bound_file(source['path'], source['sha256'])


def video_available(job, sid, exp):
    root = exp / job['id']
    path = root / (sid + '.mp4')
    audit_path = root / (sid + '.audit.json')
    audit = record(audit_path)
    need(audit['complete'] is True and audit['id'] == sid, 'incomplete/wrong video case')
    need(audit['protocol'] == 'h3_caption_image_infer_v1' and audit['mode'] == 'video_generation', 'wrong pure-video protocol')
    media = audit['media']
    need(media['verified'] is True and media['complete_decode'] is True, 'full decode pending')
    need((media['frames'], media['width'], media['height'], media['fps'], media['video_streams'], media['audio_streams']) == (124, 704, 1280, 24, 1, 0), 'wrong media geometry/audio')
    source_file(audit['video'], path)
    sampling = audit['sampling']
    need(sampling['video_scheduler_updates'] == 50 and sampling['target_audio_generated'] is False, 'wrong video sampling')
    guard = audit['audio_guard']
    need(set(guard['calls']) >= {'audio_patch', 'audio_head', 'audio_reference_encode'} and all(v == 0 for v in guard['calls'].values()), 'audio guard calls')
    need(guard['audio_vae_is_none'] is True and guard['scheduler_audio_is_none'] is True, 'audio loaded in pure video')
    for packed in (audit['packed'], sampling['packed']):
        need(packed['audio_pos_numel'] == 0 and packed['tag2_count'] == 0, 'audio packed tokens')
    run = record(root / 'run_started.json')
    need(run['mode'] == 'video_generation' and run['steps'] == 50 and run['seed'] == 42 and run['cfg'] == 1, 'run sampling binding')
    need(run['geometry'] == dict(height=1280, width=704, num_frames=124, fps=24), 'run geometry binding')
    need(run['model_sources']['audio_vae_loaded'] is False and run['model_sources']['files'], 'missing video model source')
    source_file(run['source_code'])
    source_file(run['wrapper_source'])
    caption_path = exp / job['caption']
    source_file(audit['sources']['caption_json'], caption_path)
    source_file(run['caption_source'], caption_path)
    captions = record(caption_path)
    rows = [r for r in captions['records'] if r['id'] == sid]
    need(len(rows) == 1, 'ambiguous caption case')
    row = rows[0]
    need(row['motion_caption'] == audit['motion_caption'] and row['caption_version'] == audit['caption_version'], 'caption body/version swapped')
    source_file(audit['sources']['image'], Path(row['image_path']))
    source_file(audit['condition_cache'])
    provenance = dict(caption_sha256=sha(caption_path), video_sha256=sha(path), audit_sha256=sha(audit_path), run_started_sha256=sha(root / 'run_started.json'))
    if job['kind'] == 'omni':
        prediction_path = exp / job['prediction']
        need(captions['raw_captions_unchanged'] is True and captions['source_prediction'] == str(prediction_path), 'raw source binding')
        bound_file(prediction_path, captions['source_prediction_sha256'])
        prediction = record(prediction_path)
        checkpoint = prediction.get('source_checkpoint', prediction.get('checkpoint'))
        need(isinstance(checkpoint, str) and Path(checkpoint).name == job['checkpoint'], 'wrong Omni checkpoint')
        predictions = [r for r in prediction['predictions'] if r['id'] == sid]
        need(len(predictions) == 1 and predictions[0]['caption'] == row['motion_caption'], 'raw predicted caption edited/swapped')
        source_file(dict(path=predictions[0]['audio_path'], sha256=predictions[0]['audio_sha256']))
        need(run['lora_source'] is None, 'LoRA used for frozen H3')
        if job['checkpoint'].startswith('checkpoint-u0003750'):
            finished = record(prediction_path.parent / 'finished.json')
            need(finished['complete'] is True and finished['exit_code'] == 0, 'final prediction not finished')
        else:
            need(prediction['updates'] == 3000 and len(prediction['checkpoint_adapter_sha256']) == 64, 'u3000 source verification missing')
        provenance.update(prediction_sha256=sha(prediction_path), checkpoint=checkpoint)
    else:
        verified = record(root.parent / 'weight_verified.json')
        need(verified['actual_training_updates'] == 100 and verified['tensor_keys'] == 416, 'LoRA updates/key count')
        lora = run['lora_source']['file']
        need(lora['path'] == verified['lora'] and lora['sha256'] == verified['sha256'], 'LoRA receipt/run binding')
        need(len(run['lora_source']['keys']) == verified['tensor_keys'], 'LoRA run keys')
        if job['kind'] == 'purefm':
            need(lora['sha256'] == PURE_SHA, 'wrong pureFM LoRA')
        else:
            need(run['lora_source']['alpha'] == 1.0 and run['lora_source']['scale_policy'] == 'B already contains training alpha/r; no second multiplication', 'anchor LoRA scaled more than once')
            runtime = run['runtime_lora_implementation']
            need(audit['runtime_lora_implementation'] == runtime and runtime['source_sha256'] == 'c611d16c26c38e918d4c91560f7fa72e6e88ad93b43ecad1b48fd80a91deb665', 'anchor chunk source/run binding')
            need(runtime['full_sequence_retained'] is True and runtime['weights_unchanged'] is True and runtime['original_inference_sha256'] == run['source_code']['sha256'], 'anchor original inference/weights binding')
            bound_file(runtime['source'], runtime['source_sha256'])
            anchor_path = root.parent / 'training_verified_receipt.json'
            anchor = record(anchor_path)
            receipt_path = exp / job['training_report']
            bound_file(receipt_path, job['training_report_sha256'])
            receipt = record(receipt_path)
            finished = record(receipt_path.parent / 'finished.json')
            need(finished['complete'] is True and finished['full_checkpoint_reverified'] is True and finished['report'] == str(receipt_path), 'anchor independent audit terminal binding')
            need(anchor == receipt and receipt['complete'] is True and receipt['successful_updates'] == receipt['actual_training_updates'] == 100, 'anchor flat actual training receipt binding')
            need(receipt['rng_rank_count'] == 8 and len(receipt['optimizer_shards']) == len(receipt['model_shards']) == 8, 'anchor full Adam/RNG shards pending')
            for field in ('full_checkpoint_reverified', 'independent_audit', 'typed_numeric_verified', 'lora_all_ab_source_verified', 'model_and_adam_and_rng_and_scheduler_finite', 'all_file_sha256_verified', 'fm_recipe_verified'):
                need(receipt[field] is True, 'anchor complete independent state gate: ' + field)
            need((receipt['complete_state_members'], receipt['anchor_lambda'], receipt['world_size'], receipt['lora_rank'], receipt['lora_alpha'], receipt['learning_rate'], receipt['training_cfg_scale']) == (28, 0.1, 8, 32, 64, 1e-5, 1.0), 'anchor recipe/full members binding')
            need(receipt['lora'] == verified['original_lora'] and receipt['lora'] in receipt['lora_files'], 'anchor original LoRA source swapped')
            need(receipt['lora_sha256'] == finished['lora_sha256'] == lora['sha256'] and verified['full_ab_source_verified'] is True, 'anchor LoRA all A/B source binding')
            provenance.update(training_receipt_sha256=sha(anchor_path), independent_training_report=dict(path=str(receipt_path), sha256=sha(receipt_path)), state_marker_sha256=receipt['marker_sha256'])
        provenance['lora_sha256'] = lora['sha256']
    if job.get('user_feedback'):
        feedback_path = ROOT / 'H3-posttrain/results' / job['user_feedback']
        feedback = record(feedback_path)
        need(feedback['source'] == 'user_chat' and isinstance(feedback['feedback_original'], str), 'qualitative user feedback source pending')
        provenance['qualitative_feedback_source'] = dict(path=str(feedback_path), sha256=sha(feedback_path), source='user_chat', scope=feedback['scope'], not_numeric_scores=True, not_face_hand_music_sync_acceptance=True)
    return ['../' + path.name], provenance


def native_available(job, sid, exp):
    root = exp / job['id']
    path = root / 'joint' / (sid + '_' + sid + '.mp4')
    checked_path = root / (sid + '.validated.json')
    checked = record(checked_path)
    need(checked['sample_id'] == sid and checked['path'] == str(path), 'native case/path swapped')
    need(checked['full_media_decode_checked'] is True and checked['full_video_audio_decode'] is True and checked['ffmpeg_decoded_video_frames'] == 124, 'native full AV decode pending')
    videos = [s for s in checked['streams'] if s['codec_type'] == 'video']
    need(len(videos) == 1 and (videos[0]['width'], videos[0]['height'], videos[0]['decoded_frames'], videos[0]['fps']) == (704, 1280, 124, 24), 'native video geometry')
    need(any(s['codec_type'] == 'audio' for s in checked['streams']), 'native missing audio')
    bound_file(path, checked['sha256'])
    audit_path = path.with_suffix('.audit.json')
    need(checked['audit_path'] == str(audit_path), 'native audit path swapped')
    audit = record(audit_path)
    need(audit['complete'] is True and audit['input']['case']['sample_id'] == sid and audit['output_path'] == str(path), 'native audit case/output binding')
    need(audit['video_sha256'] == checked['sha256'], 'native decode/audit SHA binding')
    inputs, calls = audit['input'], audit['audit']
    need(inputs['case']['condition'] == 'joint' and calls['scheduler_updates'] == dict(video=50, audio=50), 'native joint scheduler')
    need(calls['native_audio_condition'] == 'real_reference_audio' and calls['music_mode'] == 'ref' and calls['original_joint_pipeline'] is True, 'native audio protocol')
    need(calls['full_cached_prefix_preserved'] is True and calls['original_cached_prefix_tags_preserved'] is True and calls['packed_text_tags_exact'] is True, 'native full prefix/tags')
    need(calls['text_encoder_calls'] == 0 and calls['text_encoder_loaded'] is False and calls['tokenizer_loaded'] is False and calls['processor_loaded'] is False, 'native TC loaded/calls')
    need(inputs['student_reads_target_or_caption'] is False and inputs['teacher_identity'] is False and inputs['student_zero_input'] is False and inputs['semantic_text'] is None, 'not a non-GT student')
    need(inputs['output_audio'] == 'original_mux' and audit['mux_audio_sha256'] == inputs['native_waveform_sha256'] == calls['waveform_sha256'], 'original audio mux binding')
    case = inputs['case']
    for filename, digest in [('audio_path', 'audio_sha256'), ('image_path', 'image_sha256'), ('prefix_path', 'prefix_file_sha256')]:
        source_file(dict(path=case[filename], sha256=case[digest]))
    need(case['prefix_length'] == calls['prefix_tokens'] and len(case['prefix_tags_sha256']) == 64 and len(case['prefix_sha256']) == 64, 'native prefix source binding')
    receipt_path = exp / job['receipt']
    receipt = record(receipt_path)
    need(receipt['complete'] is True and (receipt['steps'], receipt['exposures'], receipt['optimizer_states'], receipt['state_tensors']) == (2250, 18000, 402, 467), 'native complete training state')
    need(receipt['actual_three_unique_epochs'] is True and receipt['full_saved_Adam_RNG_sampler_pointer'] is True and receipt['strict_loaded_complete_MusicFM'] is True, 'native Adam/RNG/MusicFM/epochs')
    need(receipt['checkpoint_sha256'] == receipt['local_stage_actual_sha256'] == inputs['checkpoint_sha256'] == DIRECT_SHA, 'native final checkpoint swapped')
    original = Path(receipt['original_receipt'])
    need(original == exp / 'MS/music_direct_full6k_final2250_video_n4_queue_v1/final_cpu_checkpoint.json', 'native original receipt path swapped')
    bound_file(original, receipt['original_receipt_sha256'])
    original_state = record(original)
    need(original_state['complete'] is True and original_state['checkpoint_sha256'] == DIRECT_SHA, 'native original state source')
    return [path.name], dict(video_sha256=sha(path), audit_sha256=sha(audit_path), validation_sha256=sha(checked_path), receipt_sha256=sha(receipt_path), checkpoint_sha256=DIRECT_SHA)


def collect(jobs=JOBS, exp=EXP):
    updates, provenance, pending = {}, {}, {}
    for job in jobs:
        available, sources, issues = {}, {}, {}
        root = exp / job['id']
        for sid in IDS:
            try:
                need(root.is_dir() and root.resolve() == exp.resolve() / job['id'], 'actual output missing/symlink alias')
                value, source = (native_available if job['kind'] == 'native' else video_available)(job, sid, exp)
                available[sid], sources[sid] = value, source
            except (Pending, KeyError, TypeError, ValueError, OSError, IndexError) as error:
                issues[sid] = str(error)
        updates[job['id']], provenance[job['id']], pending[job['id']] = available, sources, issues
    return updates, provenance, pending


def merge(manifest, updates, provenance, jobs=JOBS):
    result = copy.deepcopy(manifest)
    need(isinstance(result.get('models'), list) and isinstance(result.get('presets'), list), 'manifest models/presets pending')
    for job in jobs:
        available = updates[job['id']]
        if not available:
            continue
        entry = next((row for row in result['models'] if row['id'] == job['id']), None)
        if entry is None:
            entry = dict(id=job['id'], metrics={})
            result['models'].insert(0, entry)
        # Only generation-owned fields change; quality flags, metrics and reviews stay intact.
        entry.update(label=job['label'], group='MS', desc=job['desc'], audio_input=job['kind'] == 'native', audio_output=job['kind'] == 'native')
        entry.setdefault('conds', {}).setdefault('joint', {}).update(available)
        entry.setdefault('generation_provenance_v6', {}).update(provenance[job['id']])
        if not any(row['name'] == job['preset'] for row in result['presets']):
            result['presets'].insert(0, dict(name=job['preset'], cond='joint', view='models', models=[job['id']] + job['baselines']))
    return result


def atomic_save(path, value):
    temporary = path.with_name(path.name + '.v6.' + str(os.getpid()) + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def publish_once(manifest_path=MANIFEST, jobs=JOBS, exp=EXP, dry_run=False):
    updates, provenance, pending = collect(jobs, exp)
    counts = {k: len(v) for k, v in updates.items()}
    status = dict(event='focused_review_v6', pid=os.getpid(), source_sha256=sha(__file__), verified_case_count=counts,
                  verified_groups=sum(v == 3 for v in counts.values()), pending=pending,
                  complete=False, public_upload=False, numeric_reviews_written=False)
    manifest = read(manifest_path)
    if not isinstance(manifest, dict):
        status['manifest_pending'] = True
        return status
    try:
        result = merge(manifest, updates, provenance, jobs)
    except (Pending, KeyError, TypeError, ValueError) as error:
        status['manifest_pending'] = str(error)
        return status
    changed = result != manifest
    if changed and not dry_run:
        # Re-read before replace, rather than clobber edits made during media hashing.
        latest = record(manifest_path)
        result = merge(latest, updates, provenance, jobs)
        result['built'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_save(manifest_path, result)
    status.update(manifest_updated=changed and not dry_run, complete=not dry_run and all(v == len(IDS) for v in counts.values()))
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--attempts', type=int, default=2880)
    args = parser.parse_args()
    with (MANIFEST.parent / 'ready_eval_review_publisher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = None
        for attempt in range(args.attempts):
            try:
                status = publish_once(dry_run=args.dry_run)
            except (Pending, OSError, ValueError, TypeError, KeyError) as error:
                status = dict(event='focused_review_v6_pending', pid=os.getpid(), complete=False, error=str(error))
            if status != previous:
                print(json.dumps(status, ensure_ascii=False), flush=True)
                if not args.dry_run:
                    atomic_save(MANIFEST.parent / 'ready_eval_review_publisher_v6_status.json', status)
                previous = status
            if args.once or args.dry_run or status['complete']:
                return
            time.sleep(args.interval)
        raise TimeoutError('Some actual videos/sources remain pending; verified cases and human reviews retained')


if __name__ == '__main__':
    main()
