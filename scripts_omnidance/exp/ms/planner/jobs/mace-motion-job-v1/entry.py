#!/usr/bin/env python3
"""Nebula entry: build the MACE Motion environment and run its inference once.

The Motion Expert needs torch 2.4 with mamba-ssm/causal-conv1d/pytorch3d, which
are compiled from source. This entry reports each build step so failures are
localizable, then runs test.py on the released test wavs.
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

OSS_ROOT = Path(os.environ.get('MACE_OSS_ROOT', '/data/oss_bucket_0/niezichao.nzc'))
VENV_TAR = Path(os.environ.get('M2D_VENV_TAR', str(OSS_ROOT / 'h3/env/cache/venv_h3_3a418017cc795bd1.tar')))
RUN_ID = os.environ.get('MACE_RUN_ID') or time.strftime('mace_motion_%Y%m%d_%H%M%S')
BUNDLE = OSS_ROOT / 'mace' / 'bundle' / 'bundle.tar'
LOCAL = Path('/checkpoint/binary') / RUN_ID
OUT = OSS_ROOT / 'mace' / 'runs' / RUN_ID
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
    started = time.time()
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
    entry['seconds'] = round(time.time() - started, 1)
    save_report()


def run(cmd, cwd=None, env=None, timeout=7200, check=True):
    log('$ ' + ' '.join(str(part) for part in cmd))
    return subprocess.run([str(part) for part in cmd], cwd=cwd, env=env, timeout=timeout,
                          check=check, text=True, capture_output=False)


def scratch():
    LOCAL.mkdir(parents=True, exist_ok=True)
    return str(LOCAL)


def venv():
    root = LOCAL / 'venv'
    if not (root / 'venv_h3/bin/python').is_file():
        root.mkdir(exist_ok=True)
        run(['tar', '-xf', str(VENV_TAR), '-C', str(root)])
    return str(root / 'venv_h3/bin/python')


def pip(python, args, timeout=7200, no_build_isolation=False):
    command = [python, '-m', 'pip', 'install', '--no-cache-dir']
    if no_build_isolation:
        command.append('--no-build-isolation')
    return run(command + args, timeout=timeout)


def restore(root, name):
    source = OSS_ROOT / 'mace' / 'env' / name
    if source.is_file() and not (root / 'venv_h3/bin/python').is_file():
        log('restore cached ' + name)
        root.mkdir(parents=True, exist_ok=True)
        run(['tar', '-xf', str(source), '-C', str(root)])
        return True
    return False


def cached_env():
    root = LOCAL / 'venv'
    restore(root, 'venv_mace_full.tar') or restore(root, 'venv_torch24.tar')
    return venv()


def save_torch():
    root = LOCAL / 'venv'
    target = OSS_ROOT / 'mace' / 'env' / 'venv_torch24.tar'
    if target.is_file():
        return str(target)
    archive = LOCAL / 'venv_torch24.tar'
    run(['tar', '-cf', str(archive), '-C', str(root), '.'])
    run(['cp', str(archive), str(target)])
    return str(target)


def save_env():
    root = LOCAL / 'venv'
    target = OSS_ROOT / 'mace' / 'env' / 'venv_mace_full.tar'
    if target.is_file():
        return str(target)
    archive = LOCAL / 'venv_mace_full.tar'
    run(['tar', '-cf', str(archive), '-C', str(root), '.'])
    run(['cp', str(archive), str(target)])
    return str(target)


def build_env():
    root = LOCAL / 'bundle'
    if not (root / 'python/bin/python3.10').is_file():
        log('restore relocatable python + site-packages bundle')
        root.mkdir(parents=True, exist_ok=True)
        run(['tar', '-xf', str(BUNDLE), '-C', str(root)])
    python = root / 'python/bin/python3.10'
    site = root / 'site-packages'
    shim = Path(__file__).resolve().parent / 'pytorch3d_shim' / 'pytorch3d'
    target = site / 'pytorch3d'
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(shim, target)
    ld = [Path('/opt/rh/gcc-toolset-13/root/usr/lib64'), site / 'torch' / 'lib'] + sorted((site / 'nvidia').glob('*/lib'))
    check = subprocess.run([str(python), '-c',
        'import torch, mamba_ssm, causal_conv1d, pytorch3d; '
        'print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), '
        'mamba_ssm.__version__, getattr(pytorch3d, \"__version__\", \"shim\"))'],
        capture_output=True, text=True, timeout=600,
        env=dict(os.environ, PYTHONPATH=str(site),
                 LD_PRELOAD=str(LOCAL / 'bundle/lib/libstdc++.so.6'),
                 LD_LIBRARY_PATH=':'.join(str(item) for item in ld) + ':' + os.environ.get('LD_LIBRARY_PATH', '')))
    if check.returncode != 0:
        raise RuntimeError('bundle import check failed: ' + check.stderr[-3000:])
    return dict(versions=check.stdout.strip(), python=str(python), site=str(site))


def assets():
    code = LOCAL / 'code'
    if not code.is_dir():
        code.mkdir()
        run(['tar', '-xzf', str(OSS_ROOT / 'mace/code/mace_motion_code.tar.gz'), '-C', str(code)])
    motion = code / 'Expert-Motion'
    if not (motion / 'data/test/wavs').is_dir():
        archive = LOCAL / 'assets.tar'
        run(['cp', str(OSS_ROOT / 'mace/data/mace_motion_extract.tar'), str(archive)])
        run(['tar', '-xf', str(archive), '-C', str(motion)])
    return str(motion)


def infer():
    motion = assets()
    root = LOCAL / 'bundle'
    python = root / 'python/bin/python3.10'
    site = root / 'site-packages'
    shim = Path(__file__).resolve().parent / 'pytorch3d_shim' / 'pytorch3d'
    target = site / 'pytorch3d'
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(shim, target)
    ld = [Path('/opt/rh/gcc-toolset-13/root/usr/lib64'), site / 'torch' / 'lib'] + sorted((site / 'nvidia').glob('*/lib'))
    env = dict(os.environ, PYTHONPATH=str(site) + ':' + motion, TOKENIZERS_PARALLELISM='false',
               LD_PRELOAD=str(LOCAL / 'bundle/lib/libstdc++.so.6'),
               LD_LIBRARY_PATH=':'.join(str(item) for item in ld) + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    log_path = LOCAL / 'infer.log'
    with log_path.open('w') as handle:
        code = subprocess.call([str(python), 'test.py', '--feature_type', 'baseline', '--batch_size', '8'],
                               cwd=motion, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=5400)
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(log_path, OUT / 'infer.log')
    if code != 0:
        raise RuntimeError('test.py exit %d: %s' % (code, log_path.read_text(errors='replace')[-3000:]))
    produced = []
    for pattern in ('eval/**/*', 'renders/**/*'):
        for path in Path(motion).glob(pattern):
            if path.is_file():
                target = OUT / 'outputs' / path.relative_to(motion)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                produced.append(str(path.relative_to(motion)))
    return produced


def main():
    try:
        step('scratch', scratch)
        step('env', build_env)
        step('infer', infer)
        REPORT['status'] = 'ok'
    except Exception as error:
        REPORT['status'] = 'fail'
        REPORT['failed'] = repr(error)
        save_report()
        raise
    save_report()


if __name__ == '__main__':
    main()
