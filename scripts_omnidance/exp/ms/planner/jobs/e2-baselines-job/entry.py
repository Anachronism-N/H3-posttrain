#!/usr/bin/env python3
"""Entry for the first-pass E2 baseline comparison."""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

JOB = Path(__file__).resolve().parent
OSS = Path('/data/oss_bucket_0/niezichao.nzc/m2d_train_v1')
VENV_TAR = Path('/data/oss_bucket_0/niezichao.nzc/h3/env/cache/venv_h3_3a418017cc795bd1.tar')
DATA_TAR = OSS / 'data/sol61-v1.tar'
RUN_ID = os.environ.get('E2_RUN_ID') or time.strftime('e2_baselines_%Y%m%d_%H%M%S')
LOCAL = Path('/checkpoint/binary') / RUN_ID
REPORT = {'run_id': RUN_ID, 'steps': {}}


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def save():
    out = OSS / 'runs' / RUN_ID
    try:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'report.json').write_text(json.dumps(REPORT, ensure_ascii=False, indent=2) + '\n')
    except Exception as error:
        log('publish failed: %r' % (error,))


def step(name, fn):
    log('=== %s ===' % name)
    started = time.time()
    try:
        result = fn()
        REPORT['steps'][name] = dict(status='ok', seconds=round(time.time() - started, 1))
    except Exception as error:
        REPORT['steps'][name] = dict(status='fail', seconds=round(time.time() - started, 1),
                                     error=repr(error))
        save()
        raise
    save()
    return result


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    venv = LOCAL / 'venv'
    python = venv / 'venv_h3/bin/python'

    def bootstrap():
        if not python.is_file():
            venv.mkdir(exist_ok=True)
            subprocess.run(['tar', '-xf', str(VENV_TAR), '-C', str(venv)], check=True)

    def copy_data():
        data = LOCAL / 'data'
        if (data / 'val.jsonl').is_file() and (data / 'audio').is_dir():
            return
        data.mkdir(parents=True, exist_ok=True)
        with tarfile.open(DATA_TAR) as tar:
            tar.extractall(data)
        if not (data / 'val.jsonl').is_file():
            subs = [p for p in data.iterdir() if p.is_dir()]
            if len(subs) == 1 and (subs[0] / 'val.jsonl').is_file():
                for child in subs[0].iterdir():
                    child.rename(data / child.name)
                subs[0].rmdir()

    step('bootstrap', bootstrap)
    step('copy_data', copy_data)

    def run_eval():
        log_path = LOCAL / 'e2.log'
        env = dict(os.environ, PYTHONPATH=str(JOB), TOKENIZERS_PARALLELISM='false',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='8',
                   E2_RUN_ID=RUN_ID)
        with log_path.open('w') as handle:
            code = subprocess.call([str(python), str(JOB / 'e2_baselines.py')],
                                   stdout=handle, stderr=subprocess.STDOUT, env=env)
        if code != 0:
            tail = log_path.read_text(errors='replace')[-3000:]
            raise RuntimeError('e2_baselines exit %d\n%s' % (code, tail))
        shutil.copyfile(log_path, OSS / 'runs' / RUN_ID / 'e2.log')

    step('generate', run_eval)
    REPORT['complete'] = True
    save()
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        REPORT['failed'] = repr(error)
        save()
        raise
