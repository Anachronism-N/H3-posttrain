"""Event-driven formal-checkpoint acceptance and three-person deployment queue."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

from music_text_pipeline import digest
from music_scale_generate import SSH


def wait_pipeline(report_path, pid):
    if report_path.is_file():
        return
    fd = os.pidfd_open(pid)
    command = (Path('/proc') / str(pid) / 'cmdline').read_bytes()
    if b'/music_text_pipeline.py' not in command and report_path.name.encode() not in command:
        raise ValueError('Declared training pipeline PID does not match the expected report')
    libc = ctypes.CDLL(None, use_errno=True)
    watch = libc.inotify_init1(0x800 | 0x80000)
    if watch < 0 or libc.inotify_add_watch(watch, str(report_path.parent).encode(), 0x8 | 0x80 | 0x100) < 0:
        raise OSError(ctypes.get_errno(), 'inotify setup')
    deadline = time.monotonic() + 6 * 3600
    try:
        while not report_path.is_file():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Training report wait deadline; not experiment completion')
            ready, _, _ = select.select([watch, fd], [], [], remaining)
            if watch in ready:
                while True:
                    try:
                        os.read(watch, 65536)
                    except BlockingIOError:
                        break
            if fd in ready and not report_path.is_file():
                raise RuntimeError('Training pipeline exited without its acceptance report')
    finally:
        os.close(fd)
        os.close(watch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pipeline_report', type=Path, required=True)
    parser.add_argument('--pipeline_pid', type=int, required=True)
    parser.add_argument('--hosts', nargs='+', required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output_version', default='v1', choices=['v1', 'v2'])
    args = parser.parse_args()
    if args.report.exists() or not args.report.parent.is_dir():
        parser.error('Require a new report with an existing parent')
    wait_pipeline(args.pipeline_report, args.pipeline_pid)
    pipeline = json.loads(args.pipeline_report.read_text())
    if not pipeline['complete']:
        raise ValueError('Training pipeline did not pass its acceptance gates')
    exp = args.root / 'logs_omnidance/exp'
    summary_path = Path(pipeline.get('formal_summary', str(exp / 'MS/music_text_train1800_neutral_v1/finished.json')))
    summary = json.loads(summary_path.read_text())
    checkpoint = Path(summary['best_checkpoint'])
    expected_sha = summary['selected_checkpoint_deployment']['checkpoint_sha256']
    if (not summary['complete'] or digest(checkpoint) != expected_sha
            or not summary['selected_checkpoint_deployment']['exact_saved_step']):
        raise ValueError('Selected trained checkpoint does not match audited deployment state')
    jobs = []
    stages = [('trained1800', False, False), ('trained1800_zero', True, True)]
    for tag, zero, joint_only in stages:
        out = exp / 'H3AR' / ('val_musicreplace_text_' + tag + '_' + args.output_version)
        if out.exists():
            raise FileExistsError('Refuse duplicate generated experiment tree: ' + str(out))
        command = [sys.executable, str(args.root / 'scripts_omnidance/exp/ms/music_text_oracle_generate.py'),
                   '--root', str(args.root), '--checkpoint', str(checkpoint), '--out', str(out), '--hosts'] + args.hosts
        if zero:
            command += ['--zero_input']
        if joint_only:
            command += ['--joint_only']
        print(json.dumps(dict(event='generation_queue_launch', tag=tag, command=command)), flush=True)
        result = subprocess.run(command)
        jobs.append(dict(tag=tag, out=str(out), exit_code=result.returncode))
        if result.returncode:
            with args.report.open('x') as stream:
                json.dump(dict(complete=False, jobs=jobs), stream, indent=2)
            raise SystemExit(result.returncode)
    # Reuse the proven full decoder with explicitly native-fixed, empty-text logs.
    import shlex
    decoder = args.root / 'scripts_omnidance/exp/ms/music_generation_validate.py'
    tag = 'text_trained1800_' + args.output_version
    command = ['env', 'OMP_NUM_THREADS=2', '/opt/venv_eval/bin/python', str(decoder),
               '--exp_root', str(exp), '--tags', tag, '--native_tags', tag,
               '--require_empty_prompt', '--out', str(exp / 'MS' / ('music_text_trained_integrity_' + args.output_version + '.json'))]
    decoded = subprocess.run(SSH + [args.hosts[0], shlex.join(command)])
    report = dict(complete=decoded.returncode == 0, jobs=jobs, selected_checkpoint_sha256=expected_sha,
                  default_six_full_decode_exit=decoded.returncode,
                  zero_joint_clips=3, zero_decode_and_visual_review_pending=True,
                  all_new_inputs_no_semantic_text=True, musical_effect_not_yet_accepted=True)
    with args.report.open('x') as stream:
        json.dump(report, stream, indent=2)
    if decoded.returncode:
        raise SystemExit(decoded.returncode)
    print(json.dumps(dict(event='generated_default_and_zero', complete=True, videos=9)), flush=True)


if __name__ == '__main__':
    main()
