#!/usr/bin/env python3
"""MACE Appearance Expert inference on the known-good H3 venv (torch 2.8)."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

OSS_ROOT = Path(os.environ.get('MACE_OSS_ROOT', '/data/oss_bucket_0/niezichao.nzc'))
VENV_TAR = Path(os.environ.get('M2D_VENV_TAR', str(OSS_ROOT / 'h3/env/cache/venv_h3_3a418017cc795bd1.tar')))
RUN_ID = os.environ.get('MACE_RUN_ID') or time.strftime('mace_appearance_%Y%m%d_%H%M%S')
LOCAL = Path('/checkpoint/binary') / RUN_ID
OUT = OSS_ROOT / 'mace' / 'runs' / RUN_ID
CODE = OSS_ROOT / 'mace/appearance/code.tar.gz'
WEIGHTS = OSS_ROOT / 'mace/appearance/weights.tar.gz'
BASE = OSS_ROOT / 'mace/appearance/wan22_animate.tar'
REPORT = {'run_id': RUN_ID, 'steps': {}}


def log(message):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), message), flush=True)


def save_report():
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'report.json').write_text(json.dumps(REPORT, ensure_ascii=False, indent=2) + '\n')
    except Exception as error:
        log('report sync failed: %r' % error)


def step(name, fn):
    log('=== step %s ===' % name)
    entry = REPORT['steps'][name] = {'status': 'running'}
    save_report()
    try:
        entry['result'] = fn()
        entry['status'] = 'ok'
    except Exception as error:
        entry['status'] = 'fail'
        entry['error'] = repr(error)
        save_report()
        raise
    save_report()


def run(command, cwd=None, env=None, timeout=7200):
    log('$ ' + ' '.join(str(part) for part in command))
    return subprocess.run([str(part) for part in command], cwd=cwd, env=env,
                          timeout=timeout, check=True)


def scratch():
    LOCAL.mkdir(parents=True, exist_ok=True)
    return str(LOCAL)


def venv():
    root = LOCAL / 'venv'
    python = root / 'venv_h3/bin/python'
    if not python.is_file():
        root.mkdir(exist_ok=True)
        run(['tar', '-xf', str(VENV_TAR), '-C', str(root)])
    check = subprocess.run([str(python), '-c',
        'import torch, transformers, pandas, modelscope, imageio, einops, safetensors; '
        'print(transformers.__version__)'], capture_output=True, text=True)
    needs = check.returncode != 0 or not check.stdout.strip().startswith('4.57')
    if needs:
        run([str(python), '-m', 'pip', 'install', '--quiet', 'transformers==4.57.1',
             'pandas', 'modelscope', 'imageio[ffmpeg]', 'ftfy', 'sentencepiece', 'pynvml', 'protobuf'])
    return python


def assets():
    root = LOCAL / 'appearance'
    if not (root / 'Expert-Appearance/inference/appearance_infer.py').is_file():
        root.mkdir(parents=True, exist_ok=True)
        run(['tar', '-xzf', str(CODE), '-C', str(root)])
        weight_root = LOCAL / 'weight_extract'
        weight_root.mkdir(parents=True, exist_ok=True)
        run(['tar', '-xzf', str(WEIGHTS), '-C', str(weight_root)])
        shutil.copytree(weight_root / 'model', root / 'Expert-Appearance' / 'models', dirs_exist_ok=True)
    base = LOCAL / 'base'
    if not (base / 'diffusion_pytorch_model-00001-of-00004.safetensors').is_file():
        base.mkdir(parents=True, exist_ok=True)
        run(['tar', '-xf', str(BASE), '-C', str(base)])
    return root / 'Expert-Appearance', base


def infer():
    python = venv()
    repo, base = assets()
    env = dict(os.environ, PYTHONPATH=str(repo), TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1')
    csv_path = repo / 'test_data.csv'
    save_dir = LOCAL / 'outputs'
    rows = max(1, len(csv_path.read_text().strip().splitlines()) - 1)
    gpus = min(8, rows)
    processes = []
    for index in range(gpus):
        command = [str(python), 'inference/appearance_infer.py',
                   '--csv_path', str(csv_path), '--save_dir', str(save_dir),
                   '--base_model_dir', str(base),
                   '--adapter_path', str(repo / 'models/train/Adapter.safetensors'),
                   '--lora_path', str(repo / 'models/train/LoRA.safetensors'),
                   '--start-index', str(index), '--end-index', str(index + 1), '--overwrite']
        log_path = LOCAL / ('infer_%d.log' % index)
        handle = log_path.open('w')
        process = subprocess.Popen(command, cwd=repo, env=dict(env, CUDA_VISIBLE_DEVICES=str(index)),
                                   stdout=handle, stderr=subprocess.STDOUT)
        processes.append((index, process, handle, log_path))
    failures = []
    for index, process, handle, log_path in processes:
        code = process.wait()
        handle.close()
        OUT.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(log_path, OUT / ('infer_%d.log' % index))
        if code != 0:
            failures.append((index, code, log_path.read_text(errors='replace')[-1500:]))
    produced = []
    for path in Path(save_dir).rglob('*'):
        if path.is_file():
            target = OUT / 'outputs' / path.relative_to(save_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            produced.append(str(path.relative_to(save_dir)))
    if failures:
        raise RuntimeError('inference failures: %r' % failures[:2])
    return produced


def main():
    step('scratch', scratch)
    step('infer', infer)
    REPORT['status'] = 'ok'
    save_report()


if __name__ == '__main__':
    main()
