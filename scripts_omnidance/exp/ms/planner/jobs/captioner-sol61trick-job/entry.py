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
import threading
import time
from pathlib import Path

JOB = Path(__file__).resolve().parent
OSS_ROOT = Path(os.environ.get('M2D_OSS_ROOT', '/data/oss_bucket_0/niezichao.nzc/m2d_data/mirror'))
VENV_TAR = Path(os.environ.get('M2D_VENV_TAR', '/data/oss_bucket_0/niezichao.nzc/m2d_data/env/venv_h3_3a418017cc795bd1.tar'))
MODE = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get('M2D_MODE') or ((JOB / 'mode.txt').read_text().strip() if (JOB / 'mode.txt').is_file() else 'probe')).strip()
MODE_FAMILY = 'sol61' if MODE.startswith('sol61') else MODE
RUN_ID = (os.environ.get('M2D_RUN_ID') or os.environ.get('RUN_ID')
          or (MODE + '_' + time.strftime('%Y%m%d_%H%M%S')))
RANK = int(os.environ.get('RANK', '0') or 0)
LOCAL = None
REPORT = {'mode': MODE, 'run_id': RUN_ID, 'started': time.strftime('%Y-%m-%d %H:%M:%S %z'), 'steps': {}}


def shared_run_id():
    """One RUN_ID for every node of a launch.

    Nodes start seconds apart and the platform hands out no job id, so the nodes
    elect an id through a marker in the shared OSS run dir. A marker older than
    one hour belongs to a finished launch and is replaced.
    """
    if os.environ.get('M2D_RUN_ID'):
        return os.environ['M2D_RUN_ID']
    marker = OSS_ROOT / 'runs' / ('.launcher_run_id_' + MODE)
    deadline = time.time() + 600
    while time.time() < deadline:
        try:
            age = time.time() - marker.stat().st_mtime
            if age < 3600:
                adopted = marker.read_text().strip()
                if adopted:
                    log('adopted run id %s (marker age %.0fs)' % (adopted, age))
                    return adopted
        except OSError:
            pass
        try:
            fd = os.open(str(marker), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                os.write(fd, RUN_ID.encode())
            finally:
                os.close(fd)
            log('published run id %s' % RUN_ID)
            return RUN_ID
        except FileExistsError:
            try:
                if time.time() - marker.stat().st_mtime >= 3600:
                    marker.unlink()
                    continue
            except OSError:
                continue
        time.sleep(5)
    log('run id marker election timed out; using local %s' % RUN_ID)
    return RUN_ID


def log(message):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), message), flush=True)


def save_report():
    if LOCAL is not None:
        out = LOCAL / 'launcher'
        out.mkdir(parents=True, exist_ok=True)
        (out / 'report.json').write_text(json.dumps(REPORT, ensure_ascii=False, indent=2) + '\n')
    try:
        target = OSS_ROOT / 'runs' / RUN_ID
        target.mkdir(parents=True, exist_ok=True)
        if LOCAL is not None:
            for path in (LOCAL / 'launcher').rglob('*'):
                if path.is_file():
                    destination = target / path.relative_to(LOCAL / 'launcher')
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
    node = os.environ.get('NODE_RANK', os.environ.get('RANK', '0')) or '0'
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
    LOCAL = candidates[0][1] / ('captioner_jev_' + RUN_ID + '_r' + str(node))
    LOCAL.mkdir(parents=True, exist_ok=False)
    probe = Path('/checkpoint/.node_probe_r' + str(node))
    try:
        probe.write_text(os.uname().nodename)
        peers = sorted(x.name for x in Path('/checkpoint').glob('.node_probe_r*'))
        log('checkpoint_shared_probe dev=%d peers=%r' % (probe.stat().st_dev, peers))
    except OSError as error:
        log('checkpoint_shared_probe failed: %r' % (error,))
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


def resolve_model_root():
    """Mirrors were uploaded both as <root>/<name>/ and as a flattened <root>/."""
    named = OSS_ROOT / 'models' / 'Qwen3-Omni-30B-A3B-Captioner-full'
    if (named / 'model.safetensors.index.json').is_file():
        return named
    flat = OSS_ROOT / 'models'
    if (flat / 'model.safetensors.index.json').is_file():
        log('using flattened model layout at %s' % flat)
        return flat
    raise FileNotFoundError(str(named))


def copy_data():
    cfg = data_config()
    if cfg['name'] != 'balanced-v1':
        target = LOCAL / 'data'
        target.mkdir(exist_ok=True)
        archive = LOCAL / (cfg['name'] + '.tar')
        if not archive.is_file():
            shutil.copyfile(OSS_ROOT / 'data' / (cfg['name'] + '.tar'), archive)
        extract = subprocess.run(['tar', '-xf', str(archive), '-C', str(target)], capture_output=True, text=True)
        log('tar extract rc=%s stderr=%r' % (extract.returncode, extract.stderr[-2000:]))
        if extract.returncode != 0:
            raise RuntimeError('tar extract failed: ' + archive)
        # tarballs may carry a single top-level directory (sol61-v1/); flatten it
        if not (target / cfg['train']).is_file():
            subs = [x for x in target.iterdir() if x.is_dir()]
            if len(subs) == 1 and (subs[0] / cfg['train']).is_file():
                for child in subs[0].iterdir():
                    child.rename(target / child.name)
                subs[0].rmdir()
                log('flattened data from %r' % subs[0].name)
        n_top = sum(1 for _ in target.iterdir())
        n_all = sum(1 for _ in target.rglob('*'))
        wavs = sum(1 for _ in target.rglob('*.wav'))
        free = shutil.disk_usage(str(target)).free / 1024 ** 3
        log('extracted top=%d all=%d wavs=%d free_gib=%.1f archive_bytes=%d top=%r'
            % (n_top, n_all, wavs, free, archive.stat().st_size, sorted(x.name for x in target.iterdir())[:8]))
        if n_all <= 1:
            listing = subprocess.run(['tar', '-tvf', str(archive)], capture_output=True, text=True)
            log('archive listing rc=%s stderr=%r head=%r' % (listing.returncode, listing.stderr[-500:], listing.stdout[:1500]))
        archive.unlink(missing_ok=True)
        if MODE.startswith('mid2jev') and not MODE.startswith('struct'):
            lab = LOCAL / 'mid2-jev-labels.tar'
            if not lab.is_file():
                shutil.copyfile(OSS_ROOT / 'data' / 'mid2-jev-labels.tar', lab)
            subprocess.run(['tar', '-xf', str(lab), '-C', str(target)], check=True)
            lab.unlink(missing_ok=True)
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
    if MODE_FAMILY.startswith('sol61'):
        return dict(name='sol61-v1', train='train.jsonl', val='val.jsonl',
                    contract='caption_contract_nostyle.json', epochs='1', save='500', eval='500', chunk='128')
    if MODE.startswith('strat'):
        return dict(name='strat-v1', train='train.jsonl', val='val.jsonl',
                    contract='caption_contract_nostyle.json', epochs='1', save='500', eval='500', chunk='128')
    if MODE.startswith('struct'):
        return dict(name='mid2-v1', train='train.jsonl', val='val.jsonl',
                    contract='caption_contract_nostyle.json', epochs='1', save='500', eval='500', chunk='128')
    if MODE.startswith('mid2jev'):
        return dict(name='mid2-v1', train='train-jev.jsonl', val='val-jev.jsonl',
                    contract='caption_contract_nostyle.json', epochs='1', save='500', eval='500', chunk='128')
    if MODE.startswith('mid2count'):
        return dict(name='mid2-v1', train='train.jsonl', val='val.jsonl',
                    contract='caption_contract_nostyle.json', epochs='1', save='500', eval='500', chunk='128')
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
    # never reuse a directory: platform restarts repeat RUN_ID on the same node
    stamp = time.strftime('%H%M%S')
    out = LOCAL / 'out' / ('%s-attempt%s' % (RUN_ID, stamp))
    while out.exists():
        stamp = time.strftime('%H%M%S') + str(os.getpid())
        out = LOCAL / 'out' / ('%s-attempt%s' % (RUN_ID, stamp))
    TOTAL_ATTEMPTS.append(out)
    cfg = data_config()
    smoke = MODE.endswith('_smoke')
    jev = MODE.startswith('jev') or MODE.startswith('mid2jev')
    weight = '0.1' if jev else '0'
    # Nebula's pytorch launcher injects MASTER_ADDR/MASTER_PORT/ALLOCATED_RANK but not WORLD_SIZE/RANK
    # WORLD_SIZE is the node count here and NODE_RANK the node index; RANK can
    # differ from NODE_RANK on some clusters, so it is only a fallback.
    world = int(os.environ.get('WORLD_SIZE', os.environ.get('M2D_NODES', '4')) or 0)
    rank = int(os.environ.get('NODE_RANK', os.environ.get('ALLOCATED_RANK', os.environ.get('RANK', '0'))) or 0)
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    local_gpus = len([x for x in cvd.split(',') if x.strip()]) or int(os.environ.get('M2D_NPROC', '8'))
    nodes = world if world > 1 else 1
    node_rank = rank if nodes > 1 else 0
    master = os.environ.get('MASTER_ADDR', '127.0.0.1')
    port = os.environ.get('MASTER_PORT', '29500')
    log('launcher: nodes=%d node_rank=%d local_gpus=%d master=%s:%s rank_env=%r'
        % (nodes, node_rank, local_gpus, master, port, {k: os.environ.get(k) for k in ('RANK', 'NODE_RANK', 'WORLD_SIZE', 'M2D_NODES')}))
    if nodes > 1:
        launcher = ['-m', 'torch.distributed.run', '--nnodes=' + str(nodes), '--node_rank=' + str(node_rank),
                    '--master_addr=' + master, '--master_port=' + port,
                    '--nproc_per_node=' + str(local_gpus)]
    else:
        launcher = ['-m', 'torch.distributed.run', '--standalone',
                    '--nproc_per_node=' + str(local_gpus)]
    os.environ['M2D_NODES'] = str(nodes)
    global_gpus = nodes * local_gpus
    command = [str(python)] + launcher + ['--module', 'scripts_omnidance.exp.ms.music_planner_train',
               '--family', 'omni_captioner', '--model', str(model),
               '--train', str(data / cfg['train']), '--val', str(data / cfg['val']),
               '--out', str(out), '--caption-contract', str(data / cfg['contract']),
               '--epochs', cfg['epochs'], '--batch', os.environ.get('M2D_BATCH', str(global_gpus * 4)), '--micro', '1',
               '--accum', os.environ.get('M2D_ACCUM', '4'),
               '--max-updates', '2' if smoke else '0',
               '--save-every', '1' if smoke else cfg['save'], '--eval-every', '1' if smoke else cfg['eval'],
               '--rank', '32', '--alpha', '64', '--audio-last-n', '4',
               '--lora-lr', '1e-5', '--projector-lr', '5e-5', '--genre-lr', '1e-4',
               '--genre-weight', weight, '--genre-head-dim', '256',
               '--loss-chunk-size', cfg['chunk'], '--max-length', '8192',
               '--lr-schedule', 'cosine', '--warmup-updates', '20', '--seed', '42',
               '--backend', 'deepspeed', '--grad-clip', '1.']
    if jev:
        command += ['--genre-classes', str(data / 'classes.json'),
                    '--train-genre-labels', str(data / 'train-genres.jsonl'),
                    '--val-genre-labels', str(data / 'val-genres.jsonl'),
                    '--enable-jev']
    else:
        command += ['--no-enable-jev']
    if MODE.startswith('full4count') or MODE.startswith('midcount') or MODE.startswith('mid2count') or MODE.startswith('mid2jev'):
        command += ['--count-weight', '0.05', '--count-lr', '1e-4', '--count-head-dim', '128']
    if MODE.startswith('struct'):
        command += ['--count-weight', '0.05', '--count-lr', '1e-4', '--count-head-dim', '128']
    # strat: plain caption SFT only, no auxiliary heads, to preserve window-count diversity
    if MODE.startswith('mid2jev') and not MODE.startswith('struct'):
        command += ['--genre-classes', str(data / 'classes.json'),
                    '--train-genre-labels', str(data / 'train-genres.jsonl'),
                    '--val-genre-labels', str(data / 'val-genres.jsonl'),
                    '--enable-jev']
    return command, out


SYNC_STOP = threading.Event()
TOTAL_ATTEMPTS = []


def sync_out_once():
    """Push this node's training output to the shared OSS run dir.

    /checkpoint is node-local on multi-node jobs, so the ZeRO-3 shards and
    per-rank files only survive once they land in OSS. Checkpoint directories are
    uploaded only after the rank-0 trainer_state.json marks them complete.
    """
    source = LOCAL / 'out'
    if not source.is_dir():
        return {'skipped': 'no out dir'}
    target = OSS_ROOT / 'runs' / RUN_ID
    try:
        target.mkdir(parents=True, exist_ok=True)
    except Exception as error:
        log('out sync failed: %r' % (error,))
        return {'error': repr(error)}
    sent = 0
    # each launch attempt writes its own subtree; mirror them all under the run
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if relative.parts and relative.parts[0].startswith(RUN_ID + '-attempt'):
            relative = Path('attempts') / relative
        if path.suffix == '.log':
            # one shared file per run id; keep a bounded tail so progress is visible
            relative = Path('logs') / ('rank%s-' % (os.environ.get('RANK', '0') or '0') + path.name)
            destination = target / relative
            try:
                with path.open('rb') as stream:
                    stream.seek(max(0, path.stat().st_size - (2 << 20)))
                    tail = stream.read()
                (target / 'logs').mkdir(parents=True, exist_ok=True)
                (target / relative).write_bytes(tail)
            except Exception as error:
                log('log tail sync failed: %r' % (error,))
            continue
        parts = relative.parts
        if len(parts) > 2 and parts[0] == 'attempts' and parts[2].startswith('checkpoint-'):
            directory = source / parts[1] / parts[2]
            if not (directory / 'trainer_state.json').is_file():
                # adapter/ is all evaluation needs; ship it ahead of the 60GB ZeRO shards
                if not parts[3:4] == ('adapter',) or not (directory / 'adapter' / 'planner_adapter.json').is_file():
                    continue
        destination = target / relative
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.stat().st_size == path.stat().st_size:
                continue
            shutil.copyfile(path, destination)
            sent += 1
        except Exception as error:
            log('out sync item failed %r: %r' % (str(relative), error))
    if sent:
        log('out sync sent %d files to %s' % (sent, target))
    return {'sent': sent, 'target': str(target)}


def start_out_syncer():
    node = os.environ.get('NODE_RANK', os.environ.get('RANK', '0')) or '0'
    if node != '0':
        return None
    def loop():
        while not SYNC_STOP.wait(180):
            sync_out_once()
    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread


def run_training(python):
    command, out = train_command(python)
    log_path = LOCAL / 'launcher' / 'train.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(JOB), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', NCCL_DEBUG='WARN',
               CUDA_DEVICE_MAX_CONNECTIONS='1', PLANNER_SKIP_EXPERT_LORA='1', PLANNER_TOKEN_WEIGHT='1.3')
    log('$ ' + ' '.join(command))
    syncer = start_out_syncer()
    tail_path = out / 'train.log'
    with log_path.open('w') as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, text=True, env=env)
        while process.poll() is None:
            time.sleep(60)
            try:
                with log_path.open('rb') as stream:
                    stream.seek(max(0, log_path.stat().st_size - (1 << 20)))
                    tail = stream.read()
                if tail:
                    tail_path.write_bytes(tail)
            except OSError:
                pass
        code = process.wait()
    if syncer is not None:
        SYNC_STOP.set()
    sync_out_once()
    if code != 0:
        raise RuntimeError('trainer exited %d; see %s' % (code, log_path))
    return {'out': str(out), 'log': str(log_path), 'exit_code': code}


def main():
    global LOCAL
    # All ranks share one output directory; the launcher also passes M2D_RUN_ID so
    # nodes that start seconds apart cannot pick different timestamps.
    globals()['RUN_ID'] = shared_run_id()
    os.environ['M2D_RUN_ID'] = RUN_ID
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
                    'midcount_smoke', 'midcount_full', 'mid2count_smoke', 'mid2count_full',
                    'mid2jev_smoke', 'mid2jev_full', 'struct_smoke', 'struct_full',
                    'strat_smoke', 'strat_full',
                    'sol61_smoke', 'sol61_full',
                    'sol61trick_smoke', 'sol61trick_full'):
        raise ValueError('Unknown mode: ' + MODE)
    step('scratch', choose_scratch)
    venv = step('bootstrap', bootstrap_venv)
    step('copy_data', copy_data)
    step('copy_model', lambda: copy_tree(resolve_model_root(), LOCAL / 'model'))
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
