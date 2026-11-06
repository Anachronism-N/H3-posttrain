#!/usr/bin/env python3
"""Extract the frozen MusicFM 25Hz feature cache for the dense-v1 subset."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

OSS_ROOT = Path(os.environ.get('M2D_OSS_ROOT', '/data/oss_bucket_0/niezichao.nzc'))
VENV_TAR = Path(os.environ.get('M2D_VENV_TAR', str(OSS_ROOT / 'h3/env/cache/venv_h3_3a418017cc795bd1.tar')))
RUN_ID = os.environ.get('M2D_RUN_ID') or time.strftime('musicfm_cache_%Y%m%d_%H%M%S')
LOCAL = Path('/checkpoint/binary') / RUN_ID
OUT = OSS_ROOT / 'm2d_train_v1' / 'musicfm' / 'runs' / RUN_ID
ASSETS = OSS_ROOT / 'm2d_train_v1' / 'musicfm'
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


def setup():
    LOCAL.mkdir(parents=True, exist_ok=True)
    root = LOCAL / 'venv'
    python = root / 'venv_h3/bin/python'
    if not python.is_file():
        root.mkdir(exist_ok=True)
        run(['tar', '-xf', str(VENV_TAR), '-C', str(root)])
    subprocess.run([str(python), '-m', 'pip', 'install', '--quiet', 'einops', 'torchaudio==2.8.0', 'transformers==4.46.2'])
    src = LOCAL / 'musicfm_src'
    if not (src / 'musicfm/model/musicfm_25hz.py').is_file():
        src.mkdir(parents=True, exist_ok=True)
        run(['tar', '-xzf', str(ASSETS / 'musicfm-src.tar.gz'), '-C', str(src)])
    data = LOCAL / 'data'
    if not (data / 'train.jsonl').is_file():
        data.mkdir(exist_ok=True)
        run(['tar', '-xf', str(OSS_ROOT / 'm2d_train_v1/data/dense-v1.tar'), '-C', str(data)])
    code = LOCAL / 'code'
    if not (code / 'scripts_omnidance/exp/ms/music_planner_musicfm.py').is_file():
        code.mkdir(exist_ok=True)
        run(['tar', '-xzf', str(ASSETS / 'code.tar.gz'), '-C', str(code)])
    return python, src, data, code


def prepare():
    python, src, data, code = setup()
    output = LOCAL / 'cache'
    env = dict(os.environ, PYTHONPATH=str(code), TOKENIZERS_PARALLELISM='false')
    command = [str(python), '-m', 'scripts_omnidance.exp.ms.music_planner_musicfm', 'prepare',
               '--manifest', str(data / 'train.jsonl'), '--manifest', str(data / 'val.jsonl'),
               '--output-dir', str(output),
               '--musicfm-root', str(src / 'musicfm-master'),
               '--checkpoint', str(ASSETS / 'pretrained_msd.pt'),
               '--stats', str(ASSETS / 'msd_stats.json'),
               '--conformer-config', str(ASSETS / 'conformer_config.json'),
               '--device', 'cuda:0']
    prepare_log = LOCAL / 'prepare.log'
    with prepare_log.open('w') as handle:
        code_value = subprocess.call(command, cwd=code, env=env, stdout=handle,
                                     stderr=subprocess.STDOUT, timeout=7200)
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(prepare_log, OUT / 'prepare.log')
    if code_value != 0:
        raise RuntimeError('prepare exit %d: %s' % (code_value, prepare_log.read_text(errors='replace')[-2500:]))
    archive = LOCAL / 'cache-dense-v1.tar'
    run(['tar', '-cf', str(archive), '-C', str(output), '.'])
    run(['cp', str(archive), str(OUT / 'cache-dense-v1.tar')])
    return dict(bytes=archive.stat().st_size, entries=len(json.loads((output / 'index.json').read_text())['entries']))


def main():
    step('scratch', lambda: str(LOCAL.mkdir(parents=True, exist_ok=True) or LOCAL))
    step('prepare', prepare)
    REPORT['status'] = 'ok'
    save_report()


if __name__ == '__main__':
    main()
