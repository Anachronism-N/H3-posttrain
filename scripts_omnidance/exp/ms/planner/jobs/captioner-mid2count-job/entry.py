#!/usr/bin/env python3
"""Nebula entry for the Captioner SFT/JEV/loss experiments.

Modes:
  probe | jev_smoke | jev_full | sft_smoke | sft_full | loss_smoke | loss_full

Assets are read from the OSS mount, copied to node-local scratch, and verified
before the trainer is launched.  Training output is copied back under runs/.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

JOB = Path(__file__).resolve().parent
OSS_ROOT = Path(os.environ.get('M2D_OSS_ROOT', '/data/oss_bucket_0/niezichao.nzc/m2d_train_v1'))
VENV_TAR = Path(os.environ.get('M2D_VENV_TAR', '/data/oss_bucket_0/niezichao.nzc/h3/env/cache/venv_h3_3a418017cc795bd1.tar'))
MODE = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get('M2D_MODE') or ((JOB / 'mode.txt').read_text().strip() if (JOB / 'mode.txt').is_file() else 'probe')).strip()
RUN_ID = os.environ.get('M2D_RUN_ID') or (MODE + '_' + time.strftime('%Y%m%d_%H%M%S'))
LOCAL = None
REPORT = {'mode': MODE, 'run_id': RUN_ID, 'started': time.strftime('%Y-%m-%d %H:%M:%S %z'), 'steps': {}}


def log(message):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), message), flush=True)


def save_report():
    if LOCAL is not None:
        out = LOCAL / 'out'
        out.mkdir(parents=True, exist_ok=True)
        (out / 'report.json').write_text(json.dumps(REPORT, ensure_ascii=False, indent=2) + '\n')
    try:
        target = OSS_ROOT / 'runs' / RUN_ID
        target.mkdir(parents=True, exist_ok=True)
        if LOCAL is not None:
            for path in (LOCAL / 'out').rglob('*'):
                if path.is_file():
                    destination = target / path.relative_to(LOCAL / 'out')
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, destination)
    except Exception as error:
        log('OSS report sync failed: %r' % (error,))


def step(name, fn):
    log('=== step %s ===' % name)
    started = time.time()
    entry = {'status': 'running'}
    REPORT['steps'][name] = entry
    try:
        entry['result'] = fn()
        entry['status'] = 'ok'
    except Exception as error:
        entry['status'] = 'fail'
        entry['error'] = repr(error)
        log('step %s FAILED: %r' % (name, error))
        save_report()
        raise
    entry['seconds'] = round(time.time() - started, 1)
    save_report()
    return entry.get('result')


def capture(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as error:
        return 'error: %r' % (error,)


def choose_scratch():
    global LOCAL
    candidates = []
    for path in ('/checkpoint/binary', '/checkpoint', '/tmp'):
        p = Path(path)
        if p.is_dir():
            try:
                usage = shutil.disk_usage(p)
                if usage.free >= 90 * 1024 ** 3:
                    candidates.append((usage.free, p))
            except OSError:
                pass
    if not candidates:
        raise RuntimeError('No scratch >=90GiB')
    candidates.sort(reverse=True)
    LOCAL = candidates[0][1] / ('captioner_jev_' + RUN_ID)
    LOCAL.mkdir(parents=True, exist_ok=False)
    return {'chosen': str(LOCAL), 'free_gib': round(candidates[0][0] / 1024 ** 3, 1),
            'candidates': [str(p) for _, p in candidates]}


def bootstrap_venv():
    if not VENV_TAR.is_file():
        raise FileNotFoundError(str(VENV_TAR))
    venv_root = LOCAL / 'venv_root'
    venv_root.mkdir()
    log('extracting venv from %s' % VENV_TAR)
    subprocess.run(['tar', '-xf', str(VENV_TAR), '-C', str(venv_root)], check=True)
    python = venv_root / 'venv_h3' / 'bin' / 'python'
    if not python.is_file():
        raise FileNotFoundError(str(python))
    check = subprocess.run([str(python), '-c',
        'import torch, transformers, peft, accelerate, deepspeed, soundfile, numpy, scipy; '
        'print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), '
        'transformers.__version__, peft.__version__, accelerate.__version__, deepspeed.__version__)'],
        capture_output=True, text=True, timeout=300)
    if check.returncode != 0:
        raise RuntimeError('venv import failed: ' + check.stderr[-2000:])
    return {'python': str(python), 'versions': check.stdout.strip()}


def copy_tree(source, destination):
    if not source.is_dir():
        raise FileNotFoundError(str(source))
    destination.mkdir(parents=True, exist_ok=True)
    started = time.time()
    subprocess.run(['cp', '-a', str(source) + '/.', str(destination)], check=True)
    return {'source': str(source), 'destination': str(destination), 'seconds': round(time.time() - started, 1)}


def copy_data():
    cfg = data_config()
    if cfg['name'] != 'balanced-v1':
        target = LOCAL / 'data'
        target.mkdir(exist_ok=True)
        archive = LOCAL / (cfg['name'] + '.tar')
        if not archive.is_file():
            shutil.copyfile(OSS_ROOT / 'data' / (cfg['name'] + '.tar'), archive)
        subprocess.run(['tar', '-xf', str(archive), '-C', str(target)], check=True)
        archive.unlink(missing_ok=True)
        return {'archive': cfg['name'] + '.tar', 'destination': str(target)}
    return copy_tree(OSS_ROOT / 'data' / cfg['name'], LOCAL / 'data')


def verify_assets():
    model = LOCAL / 'model'
    data = LOCAL / 'data'
    shards = sorted(model.glob('model-*-of-*.safetensors'))
    if len(shards) != 16:
        raise ValueError('Expected 16 model shards, got %d' % len(shards))
    expected = json.loads((JOB / 'weight_verification.json').read_text())
    sizes = {row['path']: row['size_bytes'] for row in expected['files']}
    total = 0
    for path in sorted(model.iterdir()):
        if path.is_file():
            if path.name in sizes and path.stat().st_size != sizes[path.name]:
                raise ValueError('Model file size mismatch: ' + path.name)
            total += path.stat().st_size
    cfg = data_config()
    for name in (cfg['train'], cfg['val']):
        manifest = data / name
        if not manifest.is_file():
            raise FileNotFoundError(str(manifest))
    wavs = list((data / 'audio').glob('*.wav'))
    expected_wavs = 2400 if cfg['name'] == 'balanced-v1' else None
    if expected_wavs is not None and len(wavs) != expected_wavs:
        raise ValueError('Expected %d wav files, got %d' % (expected_wavs, len(wavs)))
    train_rows = sum(1 for line in (data / cfg['train']).open() if line.strip())
    val_rows = sum(1 for line in (data / cfg['val']).open() if line.strip())
    return {'model_bytes': total, 'shards': len(shards), 'wavs': len(wavs),
            'train': train_rows, 'val': val_rows, 'data': cfg['name']}


def data_config():
    if MODE.startswith('mid2count'):
        return dict(name='mid2-v1', train='train.jsonl', val='val.jsonl',
                    contract='caption_contract_nostyle.json', epochs='1', save='1000', eval='1000', chunk='128')
    if MODE.startswith('midcount'):
        return dict(name='mid-v1', train='train.jsonl', val='val.jsonl',
                    contract='caption_contract_nostyle.json', epochs='1', save='500', eval='500', chunk='128')
    if MODE.startswith('full4') or MODE.startswith('full4count'):
        return dict(name='full-v4', train='train.jsonl', val='val.jsonl',
                    contract='caption_contract_nostyle.json', epochs='1', save='2000', eval='2000', chunk='128')
    if MODE.startswith('jev'):
        return dict(name='balanced-v1', train='train.jsonl', val='val.jsonl',
                    contract='caption_contract.json', epochs='2', save='30', eval='30', chunk='128')
    return dict(name='full-v3', train='train.jsonl', val='val.jsonl',
                contract='caption_contract_nostyle.json', epochs='1', save='500', eval='500',
                chunk='0' if MODE.startswith('loss') else '128')


def train_command(python):
    model = LOCAL / 'model'
    data = LOCAL / 'data'
    out = LOCAL / 'out' / RUN_ID
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = data_config()
    smoke = MODE.endswith('_smoke')
    jev = MODE.startswith('jev')
    weight = '0.1' if jev else '0'
    command = [str(python), '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
               '--nproc_per_node=' + os.environ.get('M2D_NPROC', '8'), '--module', 'scripts_omnidance.exp.ms.music_planner_train',
               '--family', 'omni_captioner', '--model', str(model),
               '--train', str(data / cfg['train']), '--val', str(data / cfg['val']),
               '--out', str(out), '--caption-contract', str(data / cfg['contract']),
               '--epochs', cfg['epochs'], '--batch', os.environ.get('M2D_BATCH', '32'), '--micro', '1', '--accum', '4',
               '--max-updates', '2' if smoke else '0',
               '--save-every', '1' if smoke else cfg['save'], '--eval-every', '1' if smoke else cfg['eval'],
               '--rank', '32', '--alpha', '64', '--audio-last-n', '4',
               '--lora-lr', '1e-5', '--projector-lr', '5e-5', '--genre-lr', '1e-4',
               '--genre-weight', weight, '--genre-head-dim', '256',
               '--loss-chunk-size', cfg['chunk'], '--max-length', '8192',
               '--lr-schedule', 'cosine', '--warmup-updates', '5', '--seed', '42',
               '--backend', 'deepspeed', '--grad-clip', '1.']
    if jev:
        command += ['--genre-classes', str(data / 'classes.json'),
                    '--train-genre-labels', str(data / 'train-genres.jsonl'),
                    '--val-genre-labels', str(data / 'val-genres.jsonl'),
                    '--enable-jev']
    else:
        command += ['--no-enable-jev']
    if MODE.startswith('full4count') or MODE.startswith('midcount') or MODE.startswith('mid2count'):
        command += ['--count-weight', '0.05', '--count-lr', '1e-4', '--count-head-dim', '128']
    return command, out


def run_training(python):
    command, out = train_command(python)
    log_path = LOCAL / 'out' / 'train.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(JOB), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', NCCL_DEBUG='WARN',
               CUDA_DEVICE_MAX_CONNECTIONS='1', PLANNER_SKIP_EXPERT_LORA='1')
    log('$ ' + ' '.join(command))
    with log_path.open('w') as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, text=True, env=env)
        code = process.wait()
    if code != 0:
        raise RuntimeError('trainer exited %d; see %s' % (code, log_path))
    return {'out': str(out), 'log': str(log_path), 'exit_code': code}


def main():
    global LOCAL
    if MODE == 'probe':
        report = {
            'hostname': os.uname().nodename,
            'python': sys.version,
            'system_torch': capture([sys.executable, '-c', 'import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())']),
            'gpus': capture(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader']),
            'oss_root_exists': OSS_ROOT.is_dir(),
            'oss_data_exists': (OSS_ROOT / 'data' / 'balanced-v1' / 'train.jsonl').is_file(),
            'oss_model_exists': (OSS_ROOT / 'models' / 'Qwen3-Omni-30B-A3B-Captioner-full' / 'model.safetensors.index.json').is_file(),
            'venv_tar_exists': VENV_TAR.is_file(),
            'free_gib': {str(p): round(shutil.disk_usage(p).free / 1024 ** 3, 1) for p in ('/checkpoint/binary', '/checkpoint', '/tmp') if Path(p).is_dir()},
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if MODE not in ('jev_smoke', 'jev_full', 'sft_smoke', 'sft_full', 'loss_smoke', 'loss_full',
                    'full4_smoke', 'full4_full', 'full4count_smoke', 'full4count_full',
                    'midcount_smoke', 'midcount_full', 'mid2count_smoke', 'mid2count_full'):
        raise ValueError('Unknown mode: ' + MODE)
    step('scratch', choose_scratch)
    venv = step('bootstrap', bootstrap_venv)
    step('copy_data', copy_data)
    step('copy_model', lambda: copy_tree(OSS_ROOT / 'models' / 'Qwen3-Omni-30B-A3B-Captioner-full', LOCAL / 'model'))
    step('verify_assets', verify_assets)
    step('train', lambda: run_training(venv['python']))
    REPORT['complete'] = True
    save_report()
    print(json.dumps(REPORT, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        REPORT['failed'] = repr(error)
        save_report()
        raise
