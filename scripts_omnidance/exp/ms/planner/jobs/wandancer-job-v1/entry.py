#!/usr/bin/env python3
"""Wan-Dancer K-pop sample on 8 GPUs with xFuser sequence parallel."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

OSS = Path(os.environ.get('WD_OSS_ROOT', '/data/oss_bucket_0/niezichao.nzc/baselines'))
VENV_TAR = Path('/data/oss_bucket_0/niezichao.nzc/h3/env/cache/venv_h3_3a418017cc795bd1.tar')
RUN_ID = os.environ.get('WD_RUN_ID') or time.strftime('wandancer_%Y%m%d_%H%M%S')
LOCAL = Path('/checkpoint/binary/' + RUN_ID)
OUT = OSS / 'Wan-Dancer-runs' / RUN_ID


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    repo = LOCAL / 'Wan-Dancer'
    if not (repo / 'gen_video/gen_video_global.py').is_file():
        log('copy code')
        shutil.copytree(OSS / 'Wan-Dancer', repo, dirs_exist_ok=True)
    pipeline_file = repo / 'diffsynth/pipelines/wan_video_new.py'
    if pipeline_file.is_file():
        text = pipeline_file.read_text().replace('skip_download = dist.get_rank() != 0',
                                                 'skip_download = skip_download or dist.get_rank() != 0')
        pipeline_file.write_text(text)
    model = repo / 'models/Wan-AI/Wan-Dancer-14B'
    if not (model / 'global_model.safetensors').is_file():
        log('copy weights')
        shutil.copytree(OSS / 'Wan-Dancer-14B', model, dirs_exist_ok=True)
    venv = LOCAL / 'venv'
    python = venv / 'venv_h3' / 'bin' / 'python'
    if not python.is_file():
        venv.mkdir(exist_ok=True)
        log('extract venv')
        subprocess.run(['tar', '-xf', str(VENV_TAR), '-C', str(venv)], check=True)
        import shutil as _sh
        _sh.rmtree(venv / 'venv_h3/lib/python3.10/site-packages/xformers', ignore_errors=True)
        log('install Wan-Dancer deps')
        subprocess.run([str(python), '-m', 'pip', 'install', '--quiet',
                        'transformers==4.46.2', 'diffusers==0.34.0', 'xfuser==0.4.0',
                        'yunchang==0.5.0', 'moviepy', 'loguru', 'soxr'], check=True)
    check = subprocess.run([str(python), '-c',
        'import torch, transformers, diffusers, xfuser, yunchang, moviepy, librosa, torchvision; '
        'print(torch.__version__, torch.cuda.is_available(), transformers.__version__, diffusers.__version__)'],
        capture_output=True, text=True, timeout=300)
    if check.returncode != 0:
        raise RuntimeError('dependency check failed: ' + check.stderr[-3000:])
    log('deps ' + check.stdout.strip())
    stamp = time.strftime('%Y%m%d_%H%M%S')
    cmd = [str(python), '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8', '--nnodes=1',
           'gen_video/gen_video_global.py',
           '--seed', '0', '--image_path', 'gen_video/ref_image/3001.jpg',
           '--prompt_path', 'gen_video/prompt/kpop_global.txt',
           '--music_path', 'gen_video/music/KPopDance.WAV',
           '--output_folder', 'outputs/global_video', '--timestamp', stamp,
           '--num_inference_steps', '48', '--cfg_scale', '5']
    env = dict(os.environ, PYTHONPATH=str(repo), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    log('$ ' + ' '.join(cmd))
    log_path = LOCAL / 'wandancer.log'
    with log_path.open('w') as handle:
        code = subprocess.call(cmd, cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT)
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(log_path, OUT / 'run.log')
    if code != 0:
        tail = log_path.read_text(errors='replace')[-4000:]
        raise RuntimeError('Wan-Dancer exit %d\n%s' % (code, tail))
    videos = list((repo / 'outputs/global_video').glob('*.mp4'))
    for path in videos:
        shutil.copyfile(path, OUT / path.name)
    (OUT / 'result.json').write_text(json.dumps(dict(run_id=RUN_ID, videos=[p.name for p in videos]), indent=2) + '\n')
    log('done ' + json.dumps([p.name for p in videos]))


if __name__ == '__main__':
    main()
