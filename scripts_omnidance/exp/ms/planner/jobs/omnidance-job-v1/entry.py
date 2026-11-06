#!/usr/bin/env python3
"""Run OmniDance one-case pilot on one L20Z GPU using the cached H3 venv."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

OSS = Path(os.environ.get('OD_OSS_ROOT', '/data/oss_bucket_0/niezichao.nzc/baselines/OmniDance'))
VENV_TAR = Path('/data/oss_bucket_0/niezichao.nzc/h3/env/cache/venv_h3_3a418017cc795bd1.tar')
RUN_ID = os.environ.get('OD_RUN_ID') or time.strftime('omnidance_%Y%m%d_%H%M%S')
LOCAL = Path('/checkpoint/binary/' + RUN_ID)
OUT = OSS / 'runs' / RUN_ID


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    repo = LOCAL / 'OmniDance'
    if not (repo / 'metadata_test.csv').is_file():
        log('copy code/data')
        shutil.copytree(OSS, repo, dirs_exist_ok=True)
    if not (repo / 'model' / 'train' / 'step-20000.safetensors').is_file():
        log('copy model')
        shutil.copytree(OSS / 'model', repo / 'model', dirs_exist_ok=True)
    venv = LOCAL / 'venv'
    python = venv / 'venv_h3' / 'bin' / 'python'
    if not python.is_file():
        venv.mkdir(exist_ok=True)
        log('extract venv')
        subprocess.run(['tar', '-xf', str(VENV_TAR), '-C', str(venv)], check=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '--quiet', 'transformers==4.57.1'], check=True)
    check = subprocess.run([str(python), '-c',
        'import torch, transformers, modelscope, pandas, librosa, imageio, einops, safetensors; '
        'print(torch.__version__, torch.cuda.is_available(), transformers.__version__)'],
        capture_output=True, text=True, timeout=300)
    if check.returncode != 0:
        raise RuntimeError(check.stderr[-2000:])
    log('deps ' + check.stdout.strip())
    script = repo / 'examples/wanvideo/model_inference/Wan2.2-TI2V-5B-test.py'
    pilot = repo / 'pilot_one.py'
    text = script.read_text()
    text = text.replace('df = pd.read_csv(csv_path)', 'df = pd.read_csv(csv_path).head(4)')
    text = text.replace('output_path = f"./output"', 'output_path = f"./output_pilot"')
    pilot.write_text(text)
    env = dict(os.environ, PYTHONPATH=str(repo), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               CUDA_VISIBLE_DEVICES='0', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    log('running pilot')
    with (LOCAL / 'pilot.log').open('w') as handle:
        code = subprocess.call([str(python), 'pilot_one.py'], cwd=repo, env=env,
                               stdout=handle, stderr=subprocess.STDOUT)
    if code != 0:
        OUT.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LOCAL / 'pilot.log', OUT / 'pilot.log')
        tail = (LOCAL / 'pilot.log').read_text(errors='replace')[-4000:]
        raise RuntimeError('pilot exit %d\n%s' % (code, tail))
    OUT.mkdir(parents=True, exist_ok=True)
    produced = []
    for path in (repo / 'output_pilot').glob('*.mp4'):
        target = OUT / path.name
        shutil.copyfile(path, target)
        produced.append(str(target))
    (OUT / 'result.json').write_text(json.dumps(dict(run_id=RUN_ID, videos=produced,
        log=str(LOCAL / 'pilot.log')), indent=2) + '\n')
    log('done ' + json.dumps(produced))


if __name__ == '__main__':
    main()
