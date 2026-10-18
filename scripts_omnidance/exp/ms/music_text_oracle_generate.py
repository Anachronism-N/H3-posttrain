"""Managed three-person joint/aswap raw-text oracle, with fixed native silence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from music_scale_generate import SSH, guard_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--caps', type=Path)
    source.add_argument('--checkpoint', type=Path, help='Deploy a trained complete raw-text model, never the oracle')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--hosts', nargs='+', required=True)
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--joint_only', action='store_true', help='Three same-person teacher-label diagnostic clips only')
    parser.add_argument('--zero_input', action='store_true', help='Deploy model on zero music, original track only muxed')
    args = parser.parse_args()
    expected = 3 if args.joint_only else 6
    conditions = ['joint'] if args.joint_only else ['joint', 'aswap']
    if args.out.exists() or not args.out.parent.is_dir():
        parser.error('Require a new output tree and existing parent')
    if len(set(args.hosts)) != len(args.hosts) or not 0 <= args.gpu < 8 or len(args.hosts) > expected:
        parser.error('Require unique hosts and a valid GPU index')
    if args.zero_input and not args.checkpoint:
        parser.error('Zero music input requires a deployed model checkpoint')
    if args.caps:
        caps = json.loads(args.caps.read_text())
        if len(caps) != 6 or any(not value.get('source') for value in caps.values()):
            parser.error('Require six source-declared oracle supervision captions')
    elif not args.checkpoint.is_file():
        parser.error('Require an existing complete music text checkpoint')
    sampler = args.root / 'scripts_omnidance/exp/h3ar/sample_h3_ar.py'
    holder = args.root / 'scripts_omnidance/exp/hold_gpu.py'
    args.out.mkdir()
    (args.out / 'logs').mkdir()
    (args.out / 'cache').mkdir()
    from music_text_condition import write_new
    code_sha = hashlib.sha256(sampler.read_bytes()).hexdigest()
    write_new(args.out / 'plan.json', dict(kind='teacher_oracle_not_trained_music_model' if args.caps else 'trained_music_text_deployment',
              source_people=3, conditions=conditions, videos=expected, hosts=args.hosts,
              sampler_sha256=code_sha, supervision_caps_sha256=hashlib.sha256(args.caps.read_bytes()).hexdigest() if args.caps else None,
              checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
              native_audio='same_fixed_silence_reference_and_retake', semantic_student_prompt=' ',
              raw_text_tail_replacement=True, no_training=True, student_uses_caption=bool(args.caps), zero_music_input=args.zero_input))
    queues = [list(range(expected))[i::len(args.hosts)] for i in range(len(args.hosts))]
    guard = guard_code(str(sampler), str(holder), args.gpu)

    def run_queue(item):
        host, shards = item
        admission = subprocess.run(SSH + [host, 'python3 -c ' + shlex.quote(guard)],
                                   text=True, capture_output=True)
        write_new(args.out / 'logs' / ('admission_' + host + '.json'),
                  dict(host=host, exit_code=admission.returncode,
                       stdout=admission.stdout, stderr=admission.stderr))
        if admission.returncode:
            return [dict(host=host, shards=shards, admission_failed=True, exit_code=admission.returncode)]
        result = []
        for shard in shards:
            command = ['env', f'CUDA_VISIBLE_DEVICES={args.gpu}', 'OMP_NUM_THREADS=2',
                       'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
                       f'PYTHONPATH={args.root.parent / "DiffSynth-Studio-upstream-20260930"}',
                       '/opt/venv_h3/bin/python', str(sampler), '--out', str(args.out),
                       '--shard', str(shard), '--nshard', str(expected), '--rows', '3',
                       '--conds', ','.join(conditions), '--mode', 'bi', '--steps', '50',
                       '--height', '1280', '--width', '704', '--shift', '12',
                       '--audio_shift', '3', '--seed', '42', '--prompt_mode', 'empty',
                       '--audio_tail_mode', 'repeat_clean', '--fixed_native_silence']
            if args.caps:
                command += ['--text_hidden_oracle', str(args.caps), '--text_oracle_cache_dir', str(args.out / 'cache')]
            else:
                command += ['--music_text_ckpt', str(args.checkpoint)]
                if args.zero_input:
                    command += ['--music_text_zero_input']
            path = args.out / 'logs' / f'shard_{shard}.log'
            with path.open('x') as stream:
                stream.write(json.dumps(dict(host=host, command=command, sampler_sha256=code_sha)) + '\n')
                stream.flush()
                print(json.dumps(dict(event='launch', host=host, shard=shard)), flush=True)
                completed = subprocess.run(SSH + [host, shlex.join(command)], stdout=stream, stderr=subprocess.STDOUT)
            result.append(dict(host=host, shard=shard, exit_code=completed.returncode, log=str(path)))
            print(json.dumps(dict(event='exit', **result[-1])), flush=True)
            if completed.returncode:
                break
        return result

    with ThreadPoolExecutor(max_workers=len(args.hosts)) as executor:
        jobs = [job for queue in executor.map(run_queue, zip(args.hosts, queues)) for job in queue]
    reports = [json.loads(path.read_text()) for path in (args.out / 'cache').glob('*.json')]
    if args.checkpoint:
        import re
        reports = []
        for job in jobs:
            text = Path(job['log']).read_text()
            native = re.findall(r'^\[music text native isolation\] (.+)$', text, flags=re.MULTILINE)
            deployed = re.findall(r'^\[music text deployment\] (.+)$', text, flags=re.MULTILINE)
            if len(native) != 1 or len(deployed) != 1 or '[h3ar] done' not in text:
                raise ValueError('Missing exact trained-model deployment/native isolation success evidence')
            reports.append(dict(status='passed', native_audio=json.loads(native[0]),
                                deployment=json.loads(deployed[0]), shard=job['shard']))
    native_keys = ('waveform_sha256', 'reference_sha256', 'retake_sha256', 'mask_sha256')
    native_same = len(reports) == expected and all(
        len({report.get('native_audio', {}).get(key) for report in reports}) == 1
        and all(report.get('native_audio', {}).get(key) for report in reports) for key in native_keys)
    passed = (len(jobs) == expected and all(job['exit_code'] == 0 for job in jobs)
              and len(reports) == expected and all(report['status'] == 'passed' for report in reports) and native_same)
    final = dict(all_sampler_jobs_exit_zero=passed, expected=expected, jobs=jobs, condition_reports=reports,
                 native_silence_identical_across_all_music_and_images=native_same,
                 complete_video_decode_checked=False,
                 scope='Teacher hidden oracle, not music prediction or a trained model result' if args.caps else 'Trained shared raw-text adapter, original native music conditions disabled')
    write_new(args.out / 'jobs.json', final)
    print(json.dumps(dict(expected=expected, finished=len(jobs), oracle_conditions_passed=passed,
                          out=str(args.out))), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
