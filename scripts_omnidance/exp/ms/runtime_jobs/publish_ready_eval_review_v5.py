"""Publish each fully decoded new eval video into the existing focused review page."""
from pathlib import Path
import fcntl
import json
import os
import time
import datetime

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
EXP = ROOT / 'diff/logs_omnidance/exp'
MANIFEST = EXP / 'review_web/manifest_h3_ms_style_gt_eval3_v1.json'
IDS = ['wJPptuhclZmKAEWs', 'wJPrpFsThWAcBQIw', 'wJQTclrlOLEYYfTB']
JOBS = [
 {'id': 'MS/review_h3_purefm100_n4_chunked_v2/videos', 'label': 'H3·原MS＋图像·pureFM@100',
  'description': 'N5完成100更新后的H3 LoRA；推理LoRA临时激活分块，权重和采样保持不变；同原MS＋参考图与冻结H3纯视频基线比较，50步/seed42/CFG1/704p/124帧。内部无音频；尚未人工判优。',
  'preset': '【H3动作学习@100】pureFM 与原MS冻结H3纯视频基线',
  'baseline': 'MS/h3_ms_style_eval3_gt_videos_v1'},
 {'id': 'MS/review_omni6k_u3000_n6_video_v2/videos', 'label': 'Omni30B·6k u3000→冻结H3纯视频',
  'description': '真实音乐→u3000 Omni原始caption→同图冻结H3纯视频。原始重复文本未修改；评测四轮中间权重，不是最终u3750。内部无音频，配乐外评播放。',
  'preset': '【音乐到动作caption·6k u3000】预测caption与原MS纯视频基线',
  'baseline': 'MS/h3_ms_style_eval3_gt_videos_v1'},
 {'id': 'MS/review_h3_anchor01_s100_n4_v2/videos', 'label': 'H3·原MS＋图像·anchor0.1@100',
  'description': 'N5从原始H3训练anchor0.1实际100更新且完整Adam/RNG/LoRA核验后推理；同原MS＋参考图与冻结H3及pureFM对照，50步/seed42/CFG1/704p/124帧。内部无音频；尚未人工判优。',
  'preset': '【H3动作学习@100】anchor0.1、pureFM与原MS冻结H3纯视频基线',
  'baseline': 'MS/h3_ms_style_eval3_gt_videos_v1'},
 {'id': 'MS/music_direct_full6k_final2250_video_n4_v1', 'label': '音乐direct token·完整6k三轮@2250',
  'description': 'N2实际2250更新、18000曝光、6k三轮完整Adam/RNG及MusicFM核验后，N4固定三case。原生real reference audio joint条件，原始配乐mux；非H3纯视频无音频协议。704×1280/124帧/50步；尚未人工判优。',
  'preset': '【音乐direct token·完整6k三轮@2250】最终模型与旧mean/token@100',
  'baseline': 'MS/music_direct_video_compare_n4_v1/token',
  'direct_final': True}
]

def read(path):
 try: return json.loads(path.read_text())
 except (OSError, ValueError): return None

def direct_available(root, sid):
 path = root / 'joint' / (sid + '_' + sid + '.mp4')
 checked = read(root / (sid + '.validated.json'))
 if not path.is_file() or not isinstance(checked, dict): return None
 assert checked['sample_id'] == sid and checked['path'] == str(path)
 assert checked['full_media_decode_checked'] is True and checked['full_video_audio_decode'] is True
 assert checked['ffmpeg_decoded_video_frames'] == 124
 video = [stream for stream in checked['streams'] if stream['codec_type'] == 'video']
 assert len(video) == 1 and video[0]['width'] == 704 and video[0]['height'] == 1280
 assert video[0]['decoded_frames'] == 124 and video[0]['fps'] == 24
 assert any(stream['codec_type'] == 'audio' for stream in checked['streams'])
 audit_path = path.with_suffix('.audit.json')
 assert checked['audit_path'] == str(audit_path)
 record = read(audit_path)
 if record is None: return None
 assert record['complete'] is True and record['input']['case']['sample_id'] == sid
 assert checked['sha256'] == record['video_sha256']
 assert record['audit']['scheduler_updates'] == {'video': 50, 'audio': 50}
 assert record['audit']['native_audio_condition'] == 'real_reference_audio' and record['audit']['music_mode'] == 'ref'
 assert record['input']['student_reads_target_or_caption'] is False and record['input']['teacher_identity'] is False
 assert record['input']['student_zero_input'] is False and record['input']['output_audio'] == 'original_mux'
 assert record['mux_audio_sha256'] == record['input']['native_waveform_sha256']
 receipt = read(EXP / 'MS/music_direct_full6k_final2250_video_n4_queue_v1/final_cpu_checkpoint.json')
 if receipt is None: return None
 assert receipt['complete'] is True and receipt['steps'] == 2250 and receipt['exposures'] == 18000
 assert receipt['optimizer_states'] == 402 and receipt['state_tensors'] == 467
 assert receipt['actual_three_unique_epochs'] is True and receipt['full_saved_Adam_RNG_sampler_pointer'] is True
 assert receipt['strict_loaded_complete_MusicFM'] is True
 assert receipt['checkpoint_sha256'] == receipt['local_stage_actual_sha256'] == record['input']['checkpoint_sha256']
 return [path.name]


def main():
 lock = (MANIFEST.parent / 'ready_eval_review_publisher.lock').open('a')
 fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
 previous = {}
 previous_failures = {}
 for attempt in range(2880):
  updates = {}
  audit_failures = {}
  for job in JOBS:
   root = EXP / job['id']; available = {}
   try:
    for sid in IDS:
     if job.get('direct_final'):
      value = direct_available(root, sid)
      if value is not None: available[sid] = value
      continue
     path = root / (sid + '.mp4'); audit = read(root / (sid + '.audit.json'))
     if path.is_file() and isinstance(audit, dict) and audit.get('complete') is True and audit.get('media', {}).get('complete_decode') is True:
      assert audit['id'] == sid and audit['media']['frames'] == 124
      assert audit['media']['width'] == 704 and audit['media']['height'] == 1280
      assert audit['media']['audio_streams'] == 0 and not any(audit['audio_guard']['calls'].values())
      assert audit['video']['path'] == str(path) and audit['sampling']['video_scheduler_updates'] == 50
      available[sid] = ['../' + sid + '.mp4']
   except (AssertionError, KeyError, ValueError, TypeError) as error:
    audit_failures[job['id']] = [{'error': repr(error), 'audit_gate_failed': True}]
    previous_available = previous.get(job['id'], {})
    available = {**previous_available, **available}
   updates[job['id']] = available
  if updates != previous:
   manifest = read(MANIFEST); assert isinstance(manifest, dict)
   for job in JOBS:
    available = updates[job['id']]
    if not available: continue
    entry = {'id': job['id'], 'label': job['label'], 'group': 'MS', 'desc': job['description'],
             'conds': {'joint': available}, 'metrics': {}, 'audio_input': job.get('direct_final', False), 'audio_output': job.get('direct_final', False), 'quality_accepted': False}
    position = next((i for i, row in enumerate(manifest['models']) if row['id'] == job['id']), None)
    if position is None: manifest['models'].insert(0, entry)
    else: manifest['models'][position] = entry
    preset = {'name': job['preset'], 'cond': 'joint', 'view': 'models', 'models': ([job['id'], 'MS/music_direct_video_compare_n4_v1/mean', job['baseline']] if job.get('direct_final') else [job['id'], 'MS/review_h3_purefm100_n4_chunked_v2/videos', job['baseline']] if 'anchor01' in job['id'] else [job['id'], job['baseline']])}
    if not any(row['name'] == preset['name'] for row in manifest['presets']): manifest['presets'].insert(0, preset)
   manifest['built'] = datetime.datetime.now().isoformat()
   temporary = MANIFEST.with_name(MANIFEST.name + '.ready_eval.tmp')
   with temporary.open('w') as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2); f.flush(); os.fsync(f.fileno())
   os.replace(temporary, MANIFEST)
   previous = updates
   print(json.dumps({'event': 'review_manifest_updated', 'verified_video_count': {k: len(v) for k, v in updates.items()}}, ensure_ascii=False), flush=True)
  if not audit_failures and all(len(v) == 3 for v in updates.values()):
   print(json.dumps({'event': 'all_requested_review_videos_published', 'complete': True, 'human_review_pending': True}), flush=True); return
  terminal_failures = 0
  for job in JOBS:
   root = EXP / job['id']; output = root if job.get('direct_final') else root.parent
   failures = list(audit_failures.get(job['id'], []))
   for path in output.glob('*_exit.json'):
    record = read(path)
    if record is not None and record.get('exit_code') != 0: failures.append({'path': str(path), 'exit_code': record.get('exit_code')})
   terminal_root = EXP / 'MS/music_direct_full6k_final2250_video_n4_queue_v1' if job.get('direct_final') else output
   terminal = read(terminal_root / 'finished.json')
   if terminal is not None and terminal.get('complete') is False:
    failures.append({'path': str(terminal_root / 'finished.json'), 'error': terminal.get('error')})
   if failures:
    terminal_failures += 1
    if previous_failures.get(job['id']) != failures:
     print(json.dumps({'event': 'eval_job_failed_other_jobs_continue', 'job': job['id'], 'failures': failures, 'existing_review_videos_retained': True}, ensure_ascii=False), flush=True)
     previous_failures[job['id']] = failures
  finished_jobs = sum(len(updates[job['id']]) == 3 and job['id'] not in previous_failures for job in JOBS)
  if finished_jobs + terminal_failures == len(JOBS):
   raise RuntimeError('All eval jobs terminal; review retains completed videos, failed jobs need separate recovery')
  time.sleep(30)
 raise TimeoutError('Publisher stopped without all videos; manifest retains every verified video')

if __name__ == '__main__': main()
