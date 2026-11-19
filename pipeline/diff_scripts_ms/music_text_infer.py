"""Isolated music environment executes the entire trained text adapter, never GT caption."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import torch

from music_text_adapter import MusicToH3Text, load_music_text_checkpoint


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class MusicTextEncodingProxy:
    def __init__(self, checkpoint, python='/opt/venv_music/bin/python', legacy_clip=False):
        self.legacy_clip = bool(legacy_clip)
        self.checkpoint = str(Path(checkpoint).resolve(strict=True))
        self.checkpoint_sha256 = sha256(self.checkpoint)
        self.python = python
        self.cache_key = None
        self.cache = None
        self.calls = 0

    def __call__(self, waveform, context):
        waveform = waveform.detach().cpu().float().contiguous()
        context = context.detach().cpu().float().contiguous()
        if (waveform.shape != (2, 162000) or context.shape != (5120,)
                or not torch.isfinite(waveform).all() or not torch.isfinite(context).all()):
            raise ValueError('Proxy requires shared-loader stereo32k 162000 samples and raw5120 structural context')
        key = hashlib.sha256(self.checkpoint_sha256.encode() + waveform.numpy().tobytes() + context.numpy().tobytes()).hexdigest()
        if key != self.cache_key:
            with tempfile.TemporaryDirectory(prefix='h3_music_text_') as directory:
                input_path, output_path = Path(directory) / 'input.pt', Path(directory) / 'output.pt'
                torch.save(dict(waveform=waveform, context=context), input_path)
                environment = os.environ.copy()
                environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2')
                command = [self.python, str(Path(__file__).resolve()),
                    '--checkpoint', self.checkpoint, '--checkpoint_sha256', self.checkpoint_sha256,
                    '--input', str(input_path), '--out', str(output_path)]
                if self.legacy_clip:
                    command += ['--legacy_clip']
                completed = subprocess.run(command, env=environment, capture_output=True, text=True)
                if completed.returncode:
                    raise RuntimeError('Music text inference failed: ' + completed.stderr[-8000:] + completed.stdout[-4000:])
                data = torch.load(output_path, map_location='cpu', weights_only=True)
                tail = data['tail']
                if (data['checkpoint_sha256'] != self.checkpoint_sha256 or tail.ndim != 2
                        or tail.shape[-1] != 5120 or not 1 <= len(tail) <= data['query_max']
                        or data['predicted_length'] != len(tail) or not torch.isfinite(tail).all()):
                    raise ValueError('Invalid deployed raw text output')
                self.cache = data
                self.cache_key = key
        self.calls += 1
        print('[music text deployment] ' + json.dumps(dict(call=self.calls,
              checkpoint_sha256=self.checkpoint_sha256, input_sha256=key,
              tail_shape=list(self.cache['tail'].shape), predicted_length=self.cache['predicted_length'],
              uses_ground_truth_caption=False, final_adapter_projection_executed=True,
              legacy_clip_complete_chain=self.legacy_clip,
              coordinate_policy=self.cache.get('coordinate_policy', 'native_raw_token_prediction'))), flush=True)
        return self.cache['tail']


def _no_tc_tail_output(pipe, prompt, keyframes, text_embedding, tail, report):
    if prompt is not None or text_embedding is not None or keyframes:
        raise ValueError('No-TC accepts no text at all, including blank strings, cached text or keyframes')
    # Read storage, not tokenizer/processor properties; none of these objects may exist.
    state = vars(pipe)
    modules = state.get('_modules', {})
    for name in ('text_encoder', 'processor', 'tokenizer'):
        if state.get(name) is not None or modules.get(name) is not None:
            raise ValueError('No-TC requires the text encoder, processor and tokenizer to be absent')
    if (tail.ndim != 2 or tail.shape[-1] != 5120 or len(tail) < 1
            or not torch.isfinite(tail).all()):
        raise ValueError('No-TC requires finite raw [tokens,5120] music or diagnostic teacher tokens')
    hidden = tail.to(device=pipe.device, dtype=torch.bfloat16)
    report.update(loaded=False, calls=0, semantic_text=None, processor_loaded=False,
                  tokenizer_loaded=False, prefix_length=0, tail_shape=list(hidden.shape),
                  predicted_length=len(hidden),
                  raw_sha256=hashlib.sha256(hidden.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
    print('[no tc condition] ' + json.dumps(report, allow_nan=False), flush=True)
    return dict(prompt_embeds=hidden,
                text_token_tags=torch.ones(len(hidden), dtype=torch.long, device=pipe.device))


def student_text_embedder(checkpoint, legacy_clip=False, no_text_encoder=False):
    # Imported only in the H3 environment; the subprocess uses the music environment.
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PromptEmbedder
    if no_text_encoder and legacy_clip:
        raise ValueError('No-TC does not accept the legacy CLIP coordinate diagnostic')
    if not no_text_encoder:
        from music_text_condition import prefix_length, replace_tail

    class StudentTextTailEmbedder(MiniMaxH3Unit_PromptEmbedder):
        def __init__(self):
            super().__init__()
            if no_text_encoder:
                self.onload_model_names = ()
            self.proxy = MusicTextEncodingProxy(checkpoint, legacy_clip=legacy_clip)
            self.waveform = None

        def set_waveform(self, waveform):
            if waveform.shape != (2, 162000):
                raise ValueError('Student text path requires the shared fixed-length32k waveform')
            self.waveform = waveform

        def process(self, pipe, prompt, keyframes=None, ref_blocks=None, height=None, width=None, text_embedding=None):
            if no_text_encoder:
                if prompt is not None or text_embedding is not None or keyframes or self.waveform is None:
                    raise ValueError('No-TC student requires only the bound music waveform and no text inputs')
                # This fixed vector only accommodates an old checkpoint signature; it is not TC context.
                tail = self.proxy(self.waveform, torch.zeros(5120))
                return _no_tc_tail_output(pipe, prompt, keyframes, text_embedding, tail,
                    dict(kind='legacy_checkpoint_no_tc_music_only_diagnostic',
                         context_source='zeros_legacy_checkpoint_compatibility',
                         old_model_not_trained_for_no_tc=True, uses_ground_truth_caption=False,
                         checkpoint_sha256=self.proxy.checkpoint_sha256))
            if prompt != ' ' or text_embedding is not None or keyframes or self.waveform is None or not ref_blocks:
                raise ValueError('Student must have only music and reference structure, no semantic prompt/GT tokens')
            encoded = super().process(pipe, prompt, ref_blocks=ref_blocks, height=height, width=width)
            input_ids = self.preprocess_ref_blocks(pipe, prompt, ref_blocks)[0]
            lp = prefix_length(pipe.tokenizer, prompt, input_ids)
            hidden = encoded['prompt_embeds']
            # The isolated music process shares this GPU; release the completed text encoder first.
            context = hidden[lp - 1].detach().cpu()
            pipe.load_models_to_device([])
            tail = self.proxy(self.waveform, context)
            return replace_tail(hidden[:lp], encoded['text_token_tags'][:lp], tail)

    return StudentTextTailEmbedder()


def cached_teacher_tail_embedder(cache_dir):
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PromptEmbedder

    class CachedTeacherTailEmbedder(MiniMaxH3Unit_PromptEmbedder):
        def __init__(self):
            super().__init__()
            self.onload_model_names = ()
            self.cache_dir = Path(cache_dir)
            if not self.cache_dir.is_dir():
                raise FileNotFoundError(self.cache_dir)
            self.tail = None
            self.report = None

        def set_case(self, sample_id, image_path, audio_path):
            self.tail = self.report = None
            audio_id = Path(audio_path).stem
            stem = str(sample_id) + '_' + audio_id
            if Path(stem).name != stem:
                raise ValueError('Invalid cached teacher case identifiers')
            path = self.cache_dir / (stem + '.pt')
            source_sha256 = sha256(path)
            record = torch.load(path, map_location='cpu', weights_only=True)
            metadata = record['metadata']
            case = metadata['case']
            if (case['sample_id'] != sample_id or case['audio_id'] != audio_id
                    or str(Path(case['image_path'])) != str(Path(image_path))
                    or str(Path(case['audio_path'])) != str(Path(audio_path))
                    or case.get('condition') != 'joint'):
                raise ValueError('Cached teacher tail is not bound to the current joint image/audio/SID case')
            tail = record['tail']
            if (tail.ndim != 2 or tail.shape[1] != 5120 or len(tail) < 1
                    or len(tail) != metadata['tail_length'] or not torch.isfinite(tail).all()):
                raise ValueError('Invalid cached original teacher token count or raw tail')
            digest = hashlib.sha256()
            digest.update(str(tail.dtype).encode())
            digest.update(str(tuple(tail.shape)).encode())
            digest.update(tail.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            if digest.hexdigest() != metadata['teacher_tail_sha256']:
                raise ValueError('Cached original teacher tail hash changed')
            self.tail = tail
            self.report = dict(kind='cached_teacher_tail_only_no_tc_diagnostic',
                teacher_only_diagnostic=True, music_prediction=False,
                context_source='none_teacher_tail_only_diagnostic', source_sha256=source_sha256,
                source_path=str(path), original_teacher_tokens=len(tail),
                teacher_tail_sha256=digest.hexdigest(), sample_id=sample_id,
                image_path=str(image_path), audio_path=str(audio_path), student_reads_caption=False)

        def process(self, pipe, prompt, keyframes=None, ref_blocks=None, height=None, width=None, text_embedding=None):
            if self.tail is None or self.report is None:
                raise ValueError('Bind the cached teacher tail to a current image/audio/SID case first')
            return _no_tc_tail_output(pipe, prompt, keyframes, text_embedding, self.tail, self.report.copy())

    return CachedTeacherTailEmbedder()


def layout_audit_main():
    from types import SimpleNamespace
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder
    from music_text_condition import replace_tail, write_new, tensor_sha256
    parser = argparse.ArgumentParser(description='Shared student tail and actual H3 text/packed layout audit')
    parser.add_argument('--layout_audit', action='store_true', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--reference_target', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not args.out.parent.is_dir():
        parser.error('Require a new layout audit report')
    probe = torch.load(args.probe, map_location='cpu', weights_only=True)
    prefix_record = torch.load(args.prefix, map_location='cpu', weights_only=True)
    target = torch.load(args.reference_target, map_location='cpu', weights_only=True)
    if probe['checkpoint_sha256'] != sha256(args.checkpoint):
        raise ValueError('Probe checkpoint changed')
    if not torch.equal(prefix_record['context'].float(), probe['context'].float()):
        raise ValueError('Actual empty-prefix context differs from training/deployment probe')
    proxy = MusicTextEncodingProxy(args.checkpoint)
    deployed_tail = proxy(probe['waveform'], prefix_record['context'])
    if not torch.equal(probe['tail'], deployed_tail):
        raise ValueError('Training/evaluation tail differs from actual H3 music proxy')
    expected = replace_tail(prefix_record['prefix'], prefix_record['prefix_tags'], probe['tail'])
    actual = replace_tail(prefix_record['prefix'], prefix_record['prefix_tags'], deployed_tail)
    if any(not torch.equal(expected[key], actual[key]) for key in expected):
        raise ValueError('Packed text input is not the same raw student condition')
    width, height = target['metadata']['prepared_image_size']
    ref_blocks = [dict(kind='image', latent_t=1, latent_h=height // 16, latent_w=width // 16, ref_audio_t=0),
                  dict(kind='audio', ref_audio_t=203)]
    video = torch.empty(1, 24, ((124 - 5) // 17) * 5 + 2, 1280 // 16, 704 // 16)
    audio = torch.empty(2, 32, round(124 / 24 * 40))
    builder = MiniMaxH3Unit_PackedSequenceBuilder()
    packed = []
    for condition in (expected, actual):
        packed.append(builder.process(SimpleNamespace(device='cpu'), video_latents=video, audio_latents=audio,
                      ref_blocks=ref_blocks, **condition)['packed'])
    if set(packed[0]) != set(packed[1]) or any(
            not torch.equal(packed[0][key], packed[1][key]) if torch.is_tensor(packed[0][key])
            else packed[0][key] != packed[1][key] for key in packed[0]):
        raise ValueError('Actual H3 PackedSequenceBuilder layout differs')
    report = dict(complete=True, checkpoint_sha256=sha256(args.checkpoint),
        actual_empty_tc_prefix_used=True, student_reads_caption_or_target_length=False,
        proxy_tail_exact=True, prompt_embeds_tags_exact=True, original_packed_builder_exact=True,
        prefix_length=len(prefix_record['prefix']), student_length=len(deployed_tail),
        packed_sha256={key: tensor_sha256(value) for key, value in packed[0].items() if torch.is_tensor(value)},
        reference_prepared_size=[width, height], video_shape=list(video.shape), audio_shape=list(audio.shape),
        scope='Actual frozen prefix plus shared student condition and original H3 packed layout; no DiT prediction loss or video quality claim')
    write_new(args.out, report)
    print(json.dumps(dict(event='h3_layout_audit', **report)), flush=True)


def _no_tc_manager_cpu_audit():
    import contextlib
    import copy
    import io
    import sys
    from unittest.mock import patch
    import music_text_oracle_generate as manager

    condition = dict(loaded=False, calls=0, semantic_text=None, processor_loaded=False,
        tokenizer_loaded=False, prefix_length=0, tail_shape=[3, 5120], predicted_length=3,
        raw_sha256='a' * 64, checkpoint_sha256='b' * 64,
        old_model_not_trained_for_no_tc=True, context_source='zeros_legacy_checkpoint_compatibility')
    native = dict(text_encoder_loaded=False, tokenizer_loaded=False, processor_loaded=False,
        text_encoder_calls=0, semantic_text=None, prefix_tokens=0, prompt_embeds_shape=[3, 5120],
        visual_reference_rows=6, visual_reference_sha256='c' * 64,
        native_music_bypass=False, all_target_audio_clean=True,
        waveform_sha256='d' * 64, reference_sha256='e' * 64, retake_sha256='f' * 64, mask_sha256='0' * 64)
    deployment = dict(checkpoint_sha256='b' * 64, tail_shape=[3, 5120], predicted_length=3,
        uses_ground_truth_caption=False, final_adapter_projection_executed=True)
    mutations = [
        (False, 'condition', 'calls', False), (False, 'condition', 'calls', 0.0),
        (False, 'condition', 'prefix_length', False), (False, 'condition', 'prefix_length', 0.0),
        (False, 'condition', 'predicted_length', True), (False, 'condition', 'predicted_length', 3.0),
        (False, 'condition', 'predicted_length', 0), (False, 'condition', 'raw_sha256', 'invalid'),
        (False, 'native', 'text_encoder_loaded', 0), (False, 'native', 'tokenizer_loaded', 0),
        (False, 'native', 'processor_loaded', 0), (False, 'native', 'text_encoder_calls', False),
        (False, 'native', 'text_encoder_calls', 0.0), (False, 'native', 'semantic_text', ' '),
        (False, 'native', 'prefix_tokens', False), (False, 'native', 'prefix_tokens', 0.0),
        (False, 'native', 'prompt_embeds_shape', [4, 5120]),
        (False, 'native', 'visual_reference_rows', False), (False, 'native', 'visual_reference_rows', 0),
        (False, 'native', 'visual_reference_sha256', 'invalid'),
        (False, 'native', 'native_music_bypass', True), (False, 'native', 'all_target_audio_clean', False),
        (False, 'deployment', 'checkpoint_sha256', '1' * 64),
        (False, 'deployment', 'tail_shape', [4, 5120]), (False, 'deployment', 'predicted_length', 3.0),
        (False, 'deployment', 'uses_ground_truth_caption', True),
        (False, 'deployment', 'final_adapter_projection_executed', False),
        (True, 'condition', 'original_teacher_tokens', 3.0),
        (True, 'condition', 'original_teacher_tokens', True),
        (True, 'condition', 'source_sha256', 'invalid')]
    passed_cases, rejected = [], []
    with tempfile.TemporaryDirectory(prefix='h3_no_tc_manager_cpu_') as directory:
        parent = Path(directory)
        cache = parent / 'teacher_cache'
        cache.mkdir()
        for index in range(6):
            (cache / (str(index) + '.pt')).write_bytes(b'CPU_MOCK_ORACLE_FILE')
        checkpoint = parent / 'checkpoint.pt'
        checkpoint.write_bytes(b'CPU_MOCK_COMPLETE_CHECKPOINT')
        root = Path(__file__).resolve().parents[3]
        scenarios = [(False, None, None, None), (True, None, None, None)] + mutations
        for index, (teacher, target, field, value) in enumerate(scenarios):
            records = copy.deepcopy(dict(condition=condition, native=native, deployment=deployment))
            if teacher:
                records['condition'].update(teacher_only_diagnostic=True, music_prediction=False,
                    original_teacher_tokens=3, source_sha256='2' * 64)
            if target is not None:
                records[target][field] = value
            out = parent / ('run_' + str(index))
            argv = ['manager', '--root', str(root), '--out', str(out), '--hosts', 'cpu_mock_host',
                    '--joint_only', '--no_text_encoder']
            argv += ['--teacher_tail_cache', str(cache)] if teacher else ['--checkpoint', str(checkpoint)]
            commands = []

            def fake_run(command, **kwargs):
                if kwargs.get('capture_output'):
                    return subprocess.CompletedProcess(command, 0, 'CPU_MOCK_ADMISSION', '')
                commands.append(command)
                stream = kwargs['stdout']
                stream.write('[no tc condition] ' + json.dumps(records['condition']) + '\n')
                stream.write('[no tc native isolation] ' + json.dumps(records['native']) + '\n')
                if not teacher:
                    stream.write('[music text deployment] ' + json.dumps(records['deployment']) + '\n')
                stream.write('[h3ar] done\n')
                return subprocess.CompletedProcess(command, 0)

            with patch.object(sys, 'argv', argv), patch.object(manager.subprocess, 'run', fake_run), contextlib.redirect_stdout(io.StringIO()):
                try:
                    manager.main()
                except ValueError:
                    if target is None:
                        raise
                    rejected.append(('teacher' if teacher else 'music') + ':' + target + '.' + field + ':' + type(value).__name__)
                else:
                    if target is not None:
                        raise AssertionError('Manager accepted mutated no-TC evidence: ' + target + '.' + field)
                    plan = json.loads((out / 'plan.json').read_text())
                    jobs = json.loads((out / 'jobs.json').read_text())
                    assert plan['semantic_student_prompt'] is None and plan['prefix_length'] == 0
                    assert plan['no_text_encoder'] is True and jobs['all_sampler_jobs_exit_zero'] is True
                    assert jobs['expected'] == 3 and len(commands) == 3
                    assert all('--no_text_encoder' in command[-1] and
                               ('--no_tc_teacher_cache' in command[-1]) == teacher for command in commands)
                    passed_cases.append('teacher' if teacher else 'music')
    assert len(rejected) == len(mutations)
    return dict(complete=True, remote_or_gpu_executed=False, real_native_schema_fixture=True,
        music_and_teacher_three_case_commands_checked=True, no_tc_command_flags_exact=True,
        plan_semantic_text_null_and_prefix_zero=True, no_tc_condition_and_native_log_parsing_passed=True,
        strict_true_int_and_bool_identity_checked=True, native_condition_shape_cross_check=True,
        visual_reference_rows_and_sha256_checked=True, deployment_checkpoint_shape_length_cross_check=True,
        teacher_original_tokens_and_source_sha256_checked=True, passed_source_cases=passed_cases,
        malformed_evidence_rejected=len(rejected), rejected_mutations=rejected,
        cache_extra_aswap_files_allowed=True)


def no_tc_cpu_audit_main():
    import contextlib
    import io
    from unittest.mock import patch
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PromptEmbedder, MiniMaxH3Unit_PackedSequenceBuilder
    from music_text_condition import tensor_sha256, write_new

    parser = argparse.ArgumentParser(description='CPU-only no-TC mock and actual packed-builder audit; no video-quality claim')
    parser.add_argument('--no_tc_cpu_audit', action='store_true', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not args.out.parent.is_dir():
        parser.error('Require a new CPU audit JSON report and an existing parent')

    class ForbiddenTextPipe:
        def __init__(self):
            self.device = 'cpu'
            self.text_encoder = None
            self.forbidden_calls = 0

        def __getattribute__(self, name):
            if name in ('text_encoder', 'tokenizer', 'processor', 'load_models_to_device'):
                state = object.__getattribute__(self, '__dict__')
                state['forbidden_calls'] += 1
                raise AssertionError('Forbidden TC/tokenizer/processor/model-loader access: ' + name)
            return object.__getattribute__(self, name)

    tail = torch.arange(3 * 5120, dtype=torch.float32).reshape(3, 5120) / 1024
    proxy_inputs = []

    class MockProxy:
        def __init__(self, checkpoint, legacy_clip=False):
            self.checkpoint_sha256 = 'cpu_mock_full_proxy'

        def __call__(self, waveform, context):
            assert waveform.shape == (2, 162000)
            assert context.shape == (5120,) and context.dtype == torch.float32
            assert torch.equal(context, torch.zeros(5120))
            proxy_inputs.append(dict(waveform_sha256=tensor_sha256(waveform), context_sha256=tensor_sha256(context)))
            return tail.clone()

    def forbidden(*args, **kwargs):
        raise AssertionError('Original text processing was invoked')

    pipe = ForbiddenTextPipe()
    waveform = torch.ones(2, 162000)
    visual_clean = torch.arange(24 * 4 * 6, dtype=torch.float32).reshape(1, 24, 1, 4, 6)
    audio_clean = torch.zeros(2, 32, 3)
    ref_blocks = [dict(kind='image', latent_t=1, latent_h=4, latent_w=6,
                       ref_audio_t=0, visual_clean=visual_clean),
                  dict(kind='audio', ref_audio_t=3, audio_clean=audio_clean)]
    refs_before = [tensor_sha256(visual_clean), tensor_sha256(audio_clean)]
    rejected = []
    output_logs = io.StringIO()
    with patch.object(MiniMaxH3Unit_PromptEmbedder, 'process', forbidden), patch.object(
            MiniMaxH3Unit_PromptEmbedder, 'preprocess_ref_blocks', forbidden), patch(
            __name__ + '.MusicTextEncodingProxy', MockProxy), contextlib.redirect_stdout(output_logs):
        student = student_text_embedder('cpu_mock_checkpoint', no_text_encoder=True)
        assert student.onload_model_names == ()
        student.set_waveform(waveform)
        condition = student.process(pipe, None, ref_blocks=ref_blocks)
        for field, value in [('prompt', ' '), ('prompt', ''), ('text_embedding', torch.ones(1, 5120)), ('keyframes', [object()])]:
            kwargs = dict(prompt=None, ref_blocks=ref_blocks)
            kwargs[field] = value
            try:
                student.process(pipe, **kwargs)
            except ValueError:
                rejected.append(field + ':' + str(type(value).__name__))
            else:
                raise AssertionError('No-TC accepted a forbidden text input')
        assert torch.equal(condition['prompt_embeds'], tail.bfloat16())
        assert torch.equal(condition['text_token_tags'], torch.ones(3, dtype=torch.long))
        assert len(proxy_inputs) == 1

        with tempfile.TemporaryDirectory(prefix='h3_no_tc_cpu_') as directory:
            case = dict(sample_id='cpu_sid', audio_id='cpu_audio', condition='joint',
                        image_path='/cpu_image.png', audio_path='/cpu_audio.wav')
            cache = dict(tail=tail, metadata=dict(case=case, tail_length=3,
                         teacher_tail_sha256=tensor_sha256(tail)),
                         prefix_context='FORBIDDEN_PREFIX_SENTINEL', token_ids='FORBIDDEN_TOKENIZER_SENTINEL')
            source = Path(directory) / 'cpu_sid_cpu_audio.pt'
            torch.save(cache, source)
            teacher = cached_teacher_tail_embedder(directory)
            assert teacher.onload_model_names == ()
            teacher.set_case('cpu_sid', '/cpu_image.png', '/cpu_audio.wav')
            teacher_condition = teacher.process(pipe, None, ref_blocks=ref_blocks)
            assert all(torch.equal(condition[key], teacher_condition[key]) for key in condition)
            try:
                teacher.set_case('cpu_sid', '/wrong_image.png', '/cpu_audio.wav')
            except ValueError:
                rejected.append('teacher_image_binding')
            else:
                raise AssertionError('Unbound cached teacher image was accepted')

    builder = MiniMaxH3Unit_PackedSequenceBuilder()
    video = torch.empty(1, 24, 2, 4, 6)
    audio = torch.empty(2, 32, 5)
    actual = builder.process(pipe, video_latents=video, audio_latents=audio,
                             ref_blocks=ref_blocks, **condition)['packed']
    direct = builder.process(pipe, prompt_embeds=tail.bfloat16(), text_token_tags=torch.ones(3, dtype=torch.long),
                             video_latents=video, audio_latents=audio, ref_blocks=ref_blocks)['packed']
    assert set(actual) == set(direct)
    assert all(torch.equal(actual[key], direct[key]) if torch.is_tensor(actual[key])
               else actual[key] == direct[key] for key in actual)
    assert torch.equal(actual['text_pos'], torch.arange(3))
    assert torch.equal(actual['img_pos'], torch.cat((torch.arange(3, 9), torch.arange(25, 37))))
    assert torch.equal(actual['audio_pos'], torch.arange(9, 25))
    assert torch.equal(actual['cu_seqlens'], torch.tensor([0, 37, 64], dtype=torch.int32))
    assert torch.equal(actual['token_tags'][:3], torch.ones(3, dtype=torch.long))
    assert refs_before == [tensor_sha256(visual_clean), tensor_sha256(audio_clean)]
    assert pipe.forbidden_calls == 0
    records = [json.loads(line[len('[no tc condition] '):]) for line in output_logs.getvalue().splitlines()
               if line.startswith('[no tc condition] ')]
    assert len(records) == 2
    report = dict(complete=True, cpu_only=True, gpu_started=False,
        manager_mock_audit=_no_tc_manager_cpu_audit(),
        text_encoder_tokenizer_processor_accesses=pipe.forbidden_calls,
        original_prompt_process_and_preprocess_calls=0, no_text_encoder_onload_names=[],
        zero5120_legacy_context_exact=True, proxy_calls=len(proxy_inputs), proxy_inputs=proxy_inputs,
        no_prefix_exact=True, bf16_tail_and_all_one_tags_exact=True, forbidden_inputs_rejected=rejected,
        cached_teacher_case_binding_checked=True, cached_teacher_prefix_context_and_token_ids_unused=True,
        original_packed_builder_exact=True, packed_independent_reference_positions_exact=True,
        independent_vae_reference_tensors_unchanged=True, actual_vae_encoding_executed=False,
        text_shape=list(condition['prompt_embeds'].shape),
        video_shape=list(video.shape), audio_shape=list(audio.shape),
        packed_sha256={key: tensor_sha256(value) for key, value in actual.items() if torch.is_tensor(value)},
        no_tc_conditions=records,
        scope='Strict CPU mock and original packed layout with independent VAE-reference tensors; no real music checkpoint/GPU VAE/DiT/video execution and no new-interface success claim')
    write_new(args.out, report)
    print(json.dumps(dict(event='no_tc_cpu_audit', **report), allow_nan=False), flush=True)


def main():
    if '--no_tc_cpu_audit' in __import__('sys').argv:
        return no_tc_cpu_audit_main()
    if '--layout_audit' in __import__('sys').argv:
        return layout_audit_main()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint_sha256', required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--legacy_clip', action='store_true', help='Complete old encoder/32Dadapter/CLIPhead -> one globally scaled token; diagnostic only')
    args = parser.parse_args()
    if args.out.exists() or not args.out.parent.is_dir() or sha256(args.checkpoint) != args.checkpoint_sha256:
        parser.error('Require a new output and the declared unchanged checkpoint SHA')
    data = torch.load(args.input, map_location='cpu', weights_only=True)
    if set(data) != {'waveform', 'context'}:
        raise ValueError('Inference only accepts music and structural image context, never caption/length')
    waveform, context = data['waveform'], data['context']
    if waveform.shape != (2, 162000) or context.shape != (5120,):
        raise ValueError('Invalid shared-loader waveform or raw context shape')
    if args.legacy_clip:
        from music_encoder_adapter import load_music_checkpoint
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        mean, std = checkpoint['text_mean'].float(), checkpoint['text_std'].float()
        if (mean.shape != (5120,) or std.shape != (5120,) or not torch.isfinite(mean).all()
                or not torch.isfinite(std).all() or torch.any(std <= 0)):
            raise ValueError('Old checkpoint lacks valid train-only text statistics')
        # No inverse L2 reconstruction: one global scalar matches the train pooled-vector expected norm.
        # The direction remains the old standardized CLIP coordinate; not native token equivalence.
        scale = (mean.square().sum() + std.square().sum()).sqrt()
        model = load_music_checkpoint(args.checkpoint).cuda().eval()
        with torch.no_grad():
            embedding = model(waveform.cuda()[None], effective_frames=torch.tensor([203], device='cuda'))
        tail = (embedding[0].cpu() * scale).reshape(1, 5120)
        result = dict(tail=tail, predicted_length=1, query_max=1,
                      checkpoint_sha256=args.checkpoint_sha256,
                      metadata=dict(kind='legacy_full_clip_chain_global_token', scale=float(scale),
                                    normalization_count=checkpoint.get('normalization', {}).get('count')),
                      coordinate_policy='standardized_CLIP_direction_scaled_by_train_pooled_expected_norm_not_inverse_L2',
                      uses_ground_truth_caption=False)
    else:
        model = load_music_text_checkpoint(args.checkpoint, expected_backend='musicfm')
        if not isinstance(model, MusicToH3Text):
            raise ValueError('Deployment requires a complete backbone+adapter checkpoint')
        model = model.cuda().eval()
        with torch.no_grad():
            tails, lengths = model.generate(waveform.cuda()[None], context.cuda()[None])
        result = dict(tail=tails[0].cpu(), predicted_length=int(lengths[0]),
                      query_max=model.adapter.qmax, checkpoint_sha256=args.checkpoint_sha256,
                      metadata=model.checkpoint_metadata, uses_ground_truth_caption=False)
    with args.out.open('xb') as stream:
        torch.save(result, stream)
    print(json.dumps(dict(event='deployed_raw_text', tokens=result['predicted_length'],
                          query_max=result['query_max'], checkpoint_sha256=args.checkpoint_sha256,
                          legacy_clip_complete_chain=args.legacy_clip)), flush=True)


if __name__ == '__main__':
    main()
