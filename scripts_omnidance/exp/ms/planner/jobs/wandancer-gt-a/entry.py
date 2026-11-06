#!/usr/bin/env python3
"""Wan-Dancer on benchmark10 cases with ground-truth captions as prompts (8-GPU xFuser)."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

OSS = Path(os.environ.get('WD_OSS_ROOT', '/data/oss_bucket_0/niezichao.nzc/baselines'))
VENV_TAR = Path('/data/oss_bucket_0/niezichao.nzc/h3/env/cache/venv_h3_3a418017cc795bd1.tar')
RUN_ID = os.environ.get('WD_RUN_ID') or 'wandancer_gt_bench10_v1'
LOCAL = Path('/checkpoint/binary/' + RUN_ID)
OUT = OSS / 'Wan-Dancer-runs-GT' / RUN_ID


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
    shutil.copytree(OSS / 'benchmark10_gt', LOCAL / 'benchmark10_gt', dirs_exist_ok=True)
    shutil.copyfile(Path(__file__).parent / 'batch_global.py', repo / 'batch_global.py')
    manifest = json.loads((LOCAL / 'benchmark10_gt/manifest.json').read_text())
    done = {p.name[:-4] for p in OUT.glob('*.mp4')} if OUT.is_dir() else set()
    selected = [row for row in manifest if row['id'] not in done]
    if not selected:
        log('all cases already rendered')
        return
    (LOCAL / 'benchmark10_gt/selected.json').write_text(json.dumps(selected, ensure_ascii=False) + '\n')
    log('remaining cases %r' % [row['id'] for row in selected])
    model = repo / 'models/Wan-AI/Wan-Dancer-14B'
    if not (model / 'global_model.safetensors').is_file():
        log('copy weights')
        shutil.copytree(OSS / 'Wan-Dancer-14B', model, dirs_exist_ok=True)
    venv = LOCAL / 'venv'
    python = venv / 'venv_h3' / 'bin' / 'python'
    healthy = python.is_file() and subprocess.run(
        [str(python), '-c', 'import torch, safetensors'],
        capture_output=True, text=True, timeout=300).returncode == 0
    if not healthy and venv.exists():
        log('stale/broken venv, re-extract')
        shutil.rmtree(venv, ignore_errors=True)
    if not healthy:
        venv.mkdir(exist_ok=True)
        log('extract venv')
        subprocess.run(['tar', '-xf', str(VENV_TAR), '-C', str(venv)], check=True)
        shutil.rmtree(venv / 'venv_h3/lib/python3.10/site-packages/xformers', ignore_errors=True)
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
    cmd = [str(python), '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8', '--nnodes=1',
           'batch_global.py', str(LOCAL / 'benchmark10_gt/selected.json')]
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(repo), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', WD_OUT=str(OUT))
    log('$ ' + ' '.join(cmd))
    log_path = LOCAL / 'wandancer.log'
    with log_path.open('w') as handle:
        code = subprocess.call(cmd, cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT)
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(log_path, OUT / 'run.log')
    if code != 0:
        tail = log_path.read_text(errors='replace')[-4000:]
        raise RuntimeError('Wan-Dancer exit %d\n%s' % (code, tail))
    videos = list((repo / 'outputs/benchmark10').glob('*.mp4'))
    for path in videos:
        shutil.copyfile(path, OUT / path.name)
    produced = sorted(p.name for p in OUT.glob('*.mp4'))
    (OUT / 'result.json').write_text(json.dumps(dict(run_id=RUN_ID, videos=produced), indent=2) + '\n')
    log('done ' + json.dumps([p.name for p in videos]))


if __name__ == '__main__':
    main()
