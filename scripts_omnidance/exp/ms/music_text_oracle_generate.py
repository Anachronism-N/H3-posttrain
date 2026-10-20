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
    source.add_argument('--projection', type=Path, help='Same-length projected teacher manifest; not trained music generation')
    source.add_argument('--teacher_tail_cache', type=Path, help='Existing offline teacher tails, without TC/prefix; teacher-only diagnostic')
    parser.add_argument('--no_text_encoder', action='store_true', help='Omit text encoder, tokenizer and processor entirely; no blank text input')
    parser.add_argument('--checkpoint_source', type=Path, help='Canonical completed training checkpoint when --checkpoint is a verified local byte-for-byte cache')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--hosts', nargs='+', required=True)
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--holder_path', type=Path, help='Explicit existing holder script when source tree and experiment tree differ')
    parser.add_argument('--joint_only', action='store_true', help='Three same-person teacher-label diagnostic clips only')
    parser.add_argument('--zero_input', action='store_true', help='Deploy model on zero music, original track only muxed')
    parser.add_argument('--legacy_clip', action='store_true', help='Reuse all old trained encoder/adapter/head weights as one global text token')
    args = parser.parse_args()
    expected = 3 if args.joint_only else 6
    conditions = ['joint'] if args.joint_only else ['joint', 'aswap']
    if args.out.exists() or not args.out.parent.is_dir():
        parser.error('Require a new output tree and existing parent')
    if len(set(args.hosts)) != len(args.hosts) or not 0 <= args.gpu < 8 or len(args.hosts) > expected:
        parser.error('Require unique hosts and a valid GPU index')
    if (args.zero_input or args.legacy_clip) and not args.checkpoint:
        parser.error('Zero/legacy music input requires a deployed model checkpoint')
    if args.no_text_encoder and (args.caps or args.projection or args.legacy_clip):
        parser.error('No-TC cannot combine with caption oracle, projected teacher or legacy CLIP conditions')
    if args.teacher_tail_cache and (not args.no_text_encoder or not args.joint_only
            or not args.teacher_tail_cache.is_dir()
            or len(list(args.teacher_tail_cache.glob('*.pt'))) < 3):
        parser.error('Cached teacher tail requires no_text_encoder, exactly three joint cases and existing cache files')
    if args.caps:
        caps = json.loads(args.caps.read_text())
        if len(caps) != 6 or any(not value.get('source') for value in caps.values()):
            parser.error('Require six source-declared oracle supervision captions')
    elif args.projection:
        projected = json.loads(args.projection.read_text())
        if (not args.joint_only or projected.get('complete') is not True
                or projected.get('kind') != 'projected_teacher_not_music_prediction'
                or projected.get('diagnostic_only') is not True or len(projected.get('cases', {})) != 3):
            parser.error('Projection requires a complete three-joint diagnostic manifest')
    elif args.checkpoint and not args.checkpoint.is_file():
        parser.error('Require an existing complete music text checkpoint')
    source_summary_sha256 = None
    if args.checkpoint_source:
        if not args.checkpoint or not args.no_text_encoder or not args.checkpoint_source.is_file():
            parser.error('Checkpoint cache provenance requires a noTC model and an existing canonical source')
        summary_path = args.checkpoint_source.parent / 'finished.json'
        summary = json.loads(summary_path.read_text())
        if (summary.get('complete') is not True or summary['deployment'].get('independent_inference_tail_exact') is not True
                or summary['protocol'].get('context_policy') != 'music_only_global_bias'
                or hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != summary['checkpoint_sha256']):
            parser.error('Local inference checkpoint does not match its completed canonical source SHA')
        source_summary_sha256 = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    context_policy = 'tc_prefix_context'
    if args.no_text_encoder and args.checkpoint:
        from music_text_infer import _checkpoint_context_policy
        context_policy = _checkpoint_context_policy(args.checkpoint)
    old_no_tc_model = bool(args.no_text_encoder and args.checkpoint and context_policy == 'tc_prefix_context')
    no_tc_kind = ('cached_teacher_tail_only_no_tc_diagnostic' if args.teacher_tail_cache else
                  'legacy_checkpoint_no_tc_music_only_diagnostic' if old_no_tc_model else
                  'music_only_global_bias_no_tc_deployment')
    sampler = args.root / 'scripts_omnidance/exp/h3ar/sample_h3_ar.py'
    holder = args.holder_path or args.root / 'scripts_omnidance/exp/hold_gpu.py'
    if not holder.is_file():
        parser.error('Require an existing holder path for exact process admission')
    args.out.mkdir()
    (args.out / 'logs').mkdir()
    (args.out / 'cache').mkdir()
    from music_generation_validate import atomic_write_new as write_new
    code_sha = hashlib.sha256(sampler.read_bytes()).hexdigest()
    write_new(args.out / 'plan.json', dict(kind=no_tc_kind if args.no_text_encoder else 'projected_teacher_not_music_prediction' if args.projection else 'teacher_oracle_not_trained_music_model' if args.caps else 'trained_music_text_deployment',
              source_people=3, conditions=conditions, videos=expected, hosts=args.hosts,
              sampler_sha256=code_sha, supervision_caps_sha256=hashlib.sha256(args.caps.read_bytes()).hexdigest() if args.caps else None,
              checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
              native_audio='same_fixed_silence_reference_and_retake', semantic_student_prompt=None if args.no_text_encoder else ' ',
              no_text_encoder=args.no_text_encoder, text_encoder_loaded=False if args.no_text_encoder else None,
              teacher_tail_cache=str(args.teacher_tail_cache) if args.teacher_tail_cache else None,
              old_model_not_trained_for_no_tc=old_no_tc_model, context_policy=context_policy,
              holder_path=str(holder), checkpoint_inference_path=str(args.checkpoint) if args.checkpoint else None,
              checkpoint_source=str(args.checkpoint_source or args.checkpoint) if args.checkpoint else None,
              checkpoint_source_summary_sha256=source_summary_sha256,
              prefix_length=0 if args.no_text_encoder else None,
              sampler_no_tc_arguments=(['--no_text_encoder'] + (['--no_tc_teacher_cache', str(args.teacher_tail_cache)] if args.teacher_tail_cache else [])) if args.no_text_encoder else [],
              raw_text_tail_replacement=True, no_training=True, student_uses_caption=bool(args.caps),
              zero_music_input=args.zero_input, legacy_clip_complete_chain=args.legacy_clip,
              projection_manifest=str(args.projection) if args.projection else None,
              projection_manifest_sha256=hashlib.sha256(args.projection.read_bytes()).hexdigest() if args.projection else None,
              projection_teacher_length_only=bool(args.projection)))
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
            if args.no_text_encoder:
                command += ['--no_text_encoder']
            if args.teacher_tail_cache:
                command += ['--no_tc_teacher_cache', str(args.teacher_tail_cache)]
            elif args.caps:
                command += ['--text_hidden_oracle', str(args.caps), '--text_oracle_cache_dir', str(args.out / 'cache')]
            elif args.projection:
                command += ['--text_hidden_projection', str(args.projection)]
            else:
                command += ['--music_text_ckpt', str(args.checkpoint)]
                if args.zero_input:
                    command += ['--music_text_zero_input']
                if args.legacy_clip:
                    command += ['--music_text_legacy_clip']
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
    if args.projection:
        import re
        reports = []
        for job in jobs:
            text = Path(job['log']).read_text()
            records = re.findall(r'^\[music text projection\] (.+)$', text, flags=re.MULTILINE)
            if len(records) != 1 or '[h3ar] done' not in text:
                raise ValueError('Missing complete projected-teacher layout evidence')
            record = json.loads(records[0])
            if record.get('original_packed_exact') is not True or record.get('music_prediction') is not False:
                raise ValueError('Projected teacher layout changed or is mislabeled as music prediction')
            reports.append(dict(status='passed', native_audio=record['native_audio'], projection=record))
    if args.no_text_encoder:
        import re
        reports = []
        for job in jobs:
            if 'log' not in job:
                continue
            text = Path(job['log']).read_text()
            conditions_logged = re.findall(r'^\[no tc condition\] (.+)$', text, flags=re.MULTILINE)
            native = re.findall(r'^\[no tc native isolation\] (.+)$', text, flags=re.MULTILINE)
            if len(conditions_logged) != 1 or len(native) != 1 or '[h3ar] done' not in text:
                raise ValueError('Missing complete no-TC condition/native isolation evidence')
            condition = json.loads(conditions_logged[0])
            if (condition.get('loaded') is not False or type(condition.get('calls')) is not int
                    or condition['calls'] != 0 or condition.get('semantic_text', 'missing') is not None
                    or condition.get('processor_loaded') is not False
                    or condition.get('tokenizer_loaded') is not False
                    or type(condition.get('prefix_length')) is not int or condition['prefix_length'] != 0
                    or type(condition.get('predicted_length')) is not int or condition['predicted_length'] <= 0
                    or condition.get('tail_shape') != [condition['predicted_length'], 5120]
                    or not all(type(value) is int for value in condition['tail_shape'])
                    or re.fullmatch(r'[0-9a-f]{64}', str(condition.get('raw_sha256'))) is None):
                raise ValueError('No-TC loaded/called text processing or retained invalid text/image prefix/length/hash')
            native_record = json.loads(native[0])
            if (native_record.get('text_encoder_loaded') is not False
                    or native_record.get('tokenizer_loaded') is not False
                    or native_record.get('processor_loaded') is not False
                    or type(native_record.get('text_encoder_calls')) is not int
                    or native_record['text_encoder_calls'] != 0
                    or native_record.get('semantic_text', 'missing') is not None
                    or type(native_record.get('prefix_tokens')) is not int or native_record['prefix_tokens'] != 0
                    or native_record.get('prompt_embeds_shape') != condition['tail_shape']
                    or type(native_record.get('visual_reference_rows')) is not int
                    or native_record['visual_reference_rows'] <= 0
                    or native_record.get('native_music_bypass') is not False
                    or native_record.get('all_target_audio_clean') is not True
                    or any(re.fullmatch(r'[0-9a-f]{64}', str(native_record.get(key))) is None for key in
                           ('visual_reference_sha256', 'waveform_sha256', 'reference_sha256', 'retake_sha256', 'mask_sha256'))):
                raise ValueError('No-TC native log has text processing, mismatched tail, missing visual reference or native music bypass')
            report = dict(status='passed', native_audio=native_record,
                          no_tc_condition=condition, shard=job['shard'])
            if args.teacher_tail_cache:
                if (condition.get('teacher_only_diagnostic') is not True
                        or condition.get('music_prediction') is not False
                        or type(condition.get('original_teacher_tokens')) is not int
                        or condition['original_teacher_tokens'] != condition['predicted_length']
                        or re.fullmatch(r'[0-9a-f]{64}', str(condition.get('source_sha256'))) is None):
                    raise ValueError('Cached teacher tail lacks explicit teacher-only source/length evidence')
            else:
                deployed = re.findall(r'^\[music text deployment\] (.+)$', text, flags=re.MULTILINE)
                expected_context = 'zeros_legacy_checkpoint_compatibility' if old_no_tc_model else 'none_music_only_global_bias'
                if (len(deployed) != 1 or condition.get('old_model_not_trained_for_no_tc') is not old_no_tc_model
                        or condition.get('context_source') != expected_context
                        or condition.get('kind') != no_tc_kind):
                    raise ValueError('No-TC checkpoint context policy or deployment scope differs from its metadata')
                deployment = json.loads(deployed[0])
                if (re.fullmatch(r'[0-9a-f]{64}', str(condition.get('checkpoint_sha256'))) is None
                        or deployment.get('checkpoint_sha256') != condition['checkpoint_sha256']
                        or deployment.get('tail_shape') != condition['tail_shape']
                        or type(deployment.get('predicted_length')) is not int
                        or deployment['predicted_length'] != condition['predicted_length']
                        or deployment.get('uses_ground_truth_caption') is not False
                        or deployment.get('final_adapter_projection_executed') is not True):
                    raise ValueError('No-TC deployment checkpoint/shape/length differs or GT caption/full adapter contract failed')
                report['deployment'] = deployment
            reports.append(report)
    elif args.checkpoint:
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
                 no_text_encoder=args.no_text_encoder,
                 old_model_not_trained_for_no_tc=old_no_tc_model, context_policy=context_policy,
                 scope='Cached offline teacher tail only without TC/prefix, not music prediction or a trained model result' if args.teacher_tail_cache else 'Old checkpoint music-only/no-TC zero-context diagnostic; old model not trained for no-TC, no new-interface success claim' if old_no_tc_model else 'Trained music-only global-bias model deployed without any TC/context/prefix; video integrity and quality acceptance are separate' if args.no_text_encoder else 'Same-length projected teacher, not music prediction or identity-velocity equivalence' if args.projection else 'Teacher hidden oracle, not music prediction or a trained model result' if args.caps else 'Trained shared raw-text adapter, original native music conditions disabled')
    write_new(args.out / 'jobs.json', final)
    print(json.dumps(dict(expected=expected, finished=len(jobs), oracle_conditions_passed=passed,
                          out=str(args.out))), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
