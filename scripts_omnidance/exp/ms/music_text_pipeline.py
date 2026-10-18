"""Event-driven cache audit -> real small-fit -> formal raw-token training, managed by caller."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import select
import shlex
import subprocess
import time

from music_scale_generate import SSH, guard_code


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def wait_for_target_shards(directories, manager_pid):
    marker = 'finished.json'
    missing = [path for path in directories if not (path / marker).is_file()]
    if not missing:
        return
    pidfd = __import__('os').pidfd_open(manager_pid)
    cmdline = Path('/proc') / str(manager_pid) / 'cmdline'
    if b'music_text_targets_neutral_v1_s' not in cmdline.read_bytes():
        raise ValueError('Declared cache manager PID does not match target task')
    # File-close/move/create events wake the queue. No sleep/polling/repeated cache launches.
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(0x800 | 0x80000)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'inotify_init1')
    try:
        for directory in missing:
            if not directory.is_dir():
                raise FileNotFoundError('Target task has not created its declared directory: ' + str(directory))
            if libc.inotify_add_watch(fd, str(directory).encode(), 0x8 | 0x80 | 0x100) < 0:
                raise OSError(ctypes.get_errno(), 'inotify_add_watch')
        deadline = time.monotonic() + 6 * 3600
        while missing:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Verified cache wait expired; not training completion')
            events, _, _ = select.select([fd, pidfd], [], [], remaining)
            if fd in events:
                while True:
                    try:
                        __import__('os').read(fd, 65536)
                    except BlockingIOError:
                        break
            missing = [path for path in directories if not (path / marker).is_file()]
            if pidfd in events and missing:
                raise RuntimeError('Cache manager exited with missing finished shards')
    finally:
        __import__('os').close(fd)
        __import__('os').close(pidfd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cache_manager_pid', type=int, required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or not args.report.parent.is_dir() or not 0 <= args.gpu < 8:
        parser.error('Require a new report with existing parent and valid GPU')
    root = args.root
    exp = root / 'logs_omnidance/exp/MS'
    targets = [exp / f'music_text_targets_neutral_v1_s{i}' for i in range(4)]
    features = [exp / f'music_text_features_v1_s{i}.pt' for i in range(2)]
    wait_for_target_shards(targets, args.cache_manager_pid)
    reports = [json.loads((directory / 'finished.json').read_text()) for directory in targets]
    split = json.loads((exp / 'music_scale_v4_train1800_split.json').read_text())
    expected = set(split['train'] + split['val'])
    rows = [row for report in reports for row in report['rows']]
    if (any(not report['complete'] or report['source']['teacher_transform'] != 'gender_pronouns_to_dancer_v1' for report in reports)
            or len(rows) != len(expected) or {row['sample_id'] for row in rows} != expected):
        raise ValueError('Finished target shards fail complete/unique/split/teacher protocol gate')
    for row in rows:
        if digest(row['path']) != row['file_sha256']:
            raise ValueError('Teacher-cache file digest mismatch: ' + row['sample_id'])
    print(json.dumps(dict(event='teacher_cache_audited', count=len(rows),
                         training_query_max=max(row['tail_length'] for row in rows if row['sample_id'] in set(split['train'])))), flush=True)
    guard = guard_code(str(root / 'scripts_omnidance/exp/h3ar/sample_h3_ar.py'),
                       str(root / 'scripts_omnidance/exp/hold_gpu.py'), args.gpu)
    stages = [('smallfit', 300, 50, 8, exp / 'music_text_smallfit_neutral_v1'),
              ('formal1800', 1875, 250, 0, exp / 'music_text_train1800_neutral_v1')]
    completed = []
    for stage, steps, eval_every, small_fit, out in stages:
        if out.exists():
            raise FileExistsError('Refuse duplicate training output: ' + str(out))
        log = exp / (out.name + '.log')
        command = ['env', f'CUDA_VISIBLE_DEVICES={args.gpu}', 'OMP_NUM_THREADS=2',
                   'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', '/opt/venv_music/bin/python',
                   str(root / 'scripts_omnidance/exp/ms/music_text_align.py'), '--stage', 'train',
                   '--split', str(exp / 'music_scale_v4_train1800_split.json'), '--out', str(out),
                   '--steps', str(steps), '--eval_every', str(eval_every), '--batch_size', '8',
                   '--lr', '0.0001', '--small_fit', str(small_fit), '--targets']
        command += [str(directory) for directory in targets]
        command += ['--features'] + [str(path) for path in features]
        with log.open('x') as stream:
            stream.write(json.dumps(dict(stage=stage, host=args.host, command=command)) + '\n')
            stream.flush()
            print(json.dumps(dict(event='training_launch', stage=stage, host=args.host)), flush=True)
            process = subprocess.run(SSH + [args.host, 'python3 -c ' + shlex.quote(guard) + ' && ' + shlex.join(command)],
                                     stdout=stream, stderr=subprocess.STDOUT)
        entry = dict(stage=stage, exit_code=process.returncode, log=str(log), out=str(out))
        completed.append(entry)
        if process.returncode:
            with args.report.open('x') as stream:
                json.dump(dict(complete=False, stages=completed, failure=stage), stream, indent=2)
            raise SystemExit(process.returncode)
        summary = json.loads((out / 'finished.json').read_text())
        if not summary['complete'] or not summary['real_feature_checkpoint_roundtrip_exact']:
            raise ValueError('Training finished marker fails deployment-check gate')
        entry['summary_sha256'] = digest(out / 'finished.json')
        print(json.dumps(dict(event='training_finished', stage=stage, steps=summary['steps'])), flush=True)
    with args.report.open('x') as stream:
        json.dump(dict(complete=True, stages=completed, teacher_count=len(rows),
                       generation_not_yet_evaluated=True), stream, indent=2)
    print(json.dumps(dict(event='pipeline_finished', complete=True)), flush=True)


if __name__ == '__main__':
    main()
