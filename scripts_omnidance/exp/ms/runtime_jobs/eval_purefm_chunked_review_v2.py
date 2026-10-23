"""Evaluate completed models with fixed captions; no training or caption regeneration."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
MS = ROOT / 'diff/logs_omnidance/exp/MS'
INFER = ROOT / 'H3-posttrain/scripts_omnidance/exp/ms/h3_video_infer_chunked_lora_v1.py'
INFER_SHA = 'c611d16c26c38e918d4c91560f7fa72e6e88ad93b43ecad1b48fd80a91deb665'
GT = MS / 'omni_h3_ms_style_eval3_v1/h3_gt_caption3.json'
IDS = ['wJPptuhclZmKAEWs', 'wJPrpFsThWAcBQIw', 'wJQTclrlOLEYYfTB']
HOLDER = ['/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/envs/longlive/bin/python', str(ROOT / 'diff/scripts_omnidance/exp/hold_gpu.py')]
CONFIG = {
 'purefm': {'host': '28.216.19.143', 'uuid': 'GPU-93dc1091-2ba5-13bf-9c46-e716abe867cc', 'out': MS / 'review_h3_purefm100_n4_chunked_v2'},
 'omni': {'host': '28.216.19.225', 'uuid': 'GPU-bba35f4c-6cf5-7b0b-2791-fb6db663bde5', 'out': MS / 'review_omni6k_u3000_n6_video_v2'}
}
LORA_SOURCE = str(MS / 'h3_ms_style_purefm_s100_n5_local_v6_b473f495-08c4-43ec-928f-01a167d4ec00/step-100.safetensors')
LORA_SHA = 'e5aef58b06bba1b0f8b570fb61cfe053a85d48f4d734c32e7cd923c6931749f0'

def sha(p):
 h = hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
 return h.hexdigest()

def save(p, value):
 with Path(p).open('x') as f: json.dump(value, f, ensure_ascii=False, indent=2); f.flush(); os.fsync(f.fileno())

def identity(pid):
 p = Path('/proc') / str(pid)
 env = dict(v.split('=', 1) for v in (p / 'environ').read_bytes().decode().split(chr(0)) if '=' in v)
 return {'pid': pid, 'uids': [int(v) for line in (p / 'status').read_text().splitlines() if line.startswith('Uid:') for v in line.split()[1:]], 'argv': [v.decode() for v in (p / 'cmdline').read_bytes().split(bytes([0])) if v], 'start_ticks': (p / 'stat').read_text().rsplit(')', 1)[1].split()[19], 'cuda': env.get('CUDA_VISIBLE_DEVICES', env.get('OCCUPY_GPU')), 'cgroup': (p / 'cgroup').read_text()}

def admit(cfg, out):
 cg = Path('/sys/fs/cgroup/memory')
 room = int((cg / 'memory.limit_in_bytes').read_text()) - int((cg / 'memory.usage_in_bytes').read_text())
 mem = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))) * 1024
 assert room >= 180 * 2**30 and mem >= 180 * 2**30
 for p in Path('/proc').iterdir():
  if not p.name.isdigit(): continue
  try:
   a = [v.decode() for v in (p / 'cmdline').read_bytes().split(bytes([0])) if v]
   assert str(INFER) not in a and not any(Path(x).name == 'h3_video_infer.py' for x in a), 'Existing H3 inference; no duplicate launch'
  except (FileNotFoundError, ProcessLookupError): continue
 rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
 gpu = [v.strip() for v in rows[0].split(',')]; assert gpu[0] == '0' and gpu[1] == cfg['uuid']
 reserve = Path('/tmp/od_reserve_0')
 assert not reserve.exists() or (reserve.is_file() and not reserve.is_symlink() and reserve.stat().st_uid == os.getuid() and reserve.stat().st_size == 0)
 reserve.touch()
 processes = []
 for line in subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True).splitlines():
  uuid, pid, used = [v.strip() for v in line.split(',')]
  if uuid != cfg['uuid']: continue
  v = identity(int(pid)); assert v['uids'] == [0]*4 and v['argv'] == HOLDER and v['cuda'] == '0' and v['cgroup'] == Path('/proc/self/cgroup').read_text() and float(used) <= 1100
  processes.append(v)
 assert len(processes) <= 1
 save(out / ('admission_' + str(time.time_ns()) + '.json'), {'host': cfg['host'], 'gpu': 0, 'uuid': cfg['uuid'], 'holders': processes, 'headroom_bytes': room})
 for v in processes:
  fd = os.pidfd_open(v['pid'])
  try:
   assert identity(v['pid']) == v
   signal.pidfd_send_signal(fd, signal.SIGTERM)
   assert select.select([fd], [], [], 30)[0]
  finally: os.close(fd)
 return reserve

def stage(cfg, out, name, argv):
 assert sha(INFER) == INFER_SHA
 reserve = admit(cfg, out)
 with (out / (name + '.log')).open('x') as stream:
  env = os.environ.copy(); env.update(PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', DIFFSYNTH_SKIP_DOWNLOAD='True')
  child = subprocess.Popen(['/opt/venv_h3/bin/python', '-B', '-u', str(INFER)] + argv, stdout=stream, stderr=subprocess.STDOUT, env=env)
  save(out / (name + '_live.json'), {'manager_pid': os.getpid(), 'child_pid': child.pid, 'host': cfg['host'], 'gpu': 0, 'argv': argv, 'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat()})
  print(json.dumps({'event': 'actual_video_eval_stage_started', 'job': args.job, 'stage': name, 'child_pid': child.pid}), flush=True)
  while True:
   try: code = child.wait(timeout=30); break
   except subprocess.TimeoutExpired: reserve.touch()
 save(out / (name + '_exit.json'), {'exit_code': code})
 assert code == 0, 'Stage failed; preserve partial assets'
 assert sha(INFER) == INFER_SHA

def main(args):
 cfg = CONFIG[args.job]; out = cfg['out']
 assert os.getuid() == 0 and not out.exists() and sha(INFER) == INFER_SHA
 assert args.job == 'purefm'
 assert sha(__file__) == args.manager_sha
 out.mkdir()
 rows = json.loads(GT.read_text())['records']; assert [r['id'] for r in rows] == IDS
 lora = None
 if args.job == 'purefm':
  caption = GT; cache = MS / 'h3_ms_style_eval3_gt_conditions_v1'
  assert json.loads((cache / 'run_complete.json').read_text())['completed_ids'] == IDS
  lora = Path('/opt/h3_review_purefm100_step100.safetensors')
  if not lora.exists():
   subprocess.run(['scp', '-O', '-p', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '28.216.19.137:' + LORA_SOURCE, str(lora)], check=True)
  assert sha(lora) == LORA_SHA
  from safetensors import safe_open
  with safe_open(str(lora), framework='pt', device='cpu') as f:
   keys = list(f.keys()); assert keys and all('.lora_' in k for k in keys)
   assert all((k.replace('.lora_A.', '.lora_B.') if '.lora_A.' in k else k.replace('.lora_B.', '.lora_A.')) in keys for k in keys)
  save(out / 'weight_verified.json', {'lora': str(lora), 'sha256': LORA_SHA, 'actual_training_updates': 100, 'tensor_keys': len(keys)})
 else:
  prediction_path = MS / 'n7_omni6k_u3000_eval_n6_v1/omni_prediction3_raw.json'
  predictions = json.loads(prediction_path.read_text()); assert [r['id'] for r in predictions['predictions']] == IDS and predictions['updates'] == 3000
  for row, pred in zip(rows, predictions['predictions']):
   row['motion_caption'] = pred['caption']; row['caption_version'] = 'omni6k_u3000_raw:' + sha(prediction_path)
  caption = out / 'captions.json'; save(caption, {'records': rows, 'source_prediction': str(prediction_path), 'source_prediction_sha256': sha(prediction_path), 'raw_captions_unchanged': True})
  cache = out / 'conditions'
 save(out / 'plan.json', {'job': args.job, 'ids': IDS, 'caption_json': str(caption), 'caption_sha256': sha(caption), 'cache': str(cache), 'lora': str(lora) if lora else None, 'human_quality_accepted': False, 'internal_audio': False, 'steps': 50, 'seed': 42, 'CFG': 1})
 base = ['--caption-json', str(caption), '--weights-root', '/dev/shm/h3/Ref2VA', '--limit', '3', '--steps', '50', '--cfg', '1', '--seed', '42']
 if args.job == 'omni': stage(cfg, out, 'cache', base + ['--out', str(cache), '--cache-only'])
 video = out / 'videos'
 extra = ['--lora', str(lora)] if lora else []
 stage(cfg, out, 'video', base + ['--out', str(video), '--condition-cache', str(cache)] + extra)
 report = json.loads((video / 'run_complete.json').read_text()); assert report['complete'] and report['completed_ids'] == IDS
 for sid in IDS:
  audit = json.loads((video / (sid + '.audit.json')).read_text()); assert audit['complete'] and audit['media']['complete_decode'] and audit['media']['frames'] == 124
  assert not any(audit['audio_guard']['calls'].values()) and audit['sampling']['video_scheduler_updates'] == 50
  if lora: assert report['lora_source']['file']['sha256'] == LORA_SHA
 save(out / 'finished.json', {'complete': True, 'videos': 3, 'ids': IDS, 'path': str(video), 'job': args.job, 'human_quality_accepted': False})
 print(json.dumps({'event': 'three_eval_videos_complete', 'job': args.job}), flush=True)

if __name__ == '__main__':
 parser = argparse.ArgumentParser(); parser.add_argument('--job', choices=['purefm'], required=True); parser.add_argument('--manager-sha', required=True); args = parser.parse_args(); main(args)
