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


def _checkpoint_context_policy(checkpoint):
    # Mapped tensor storage is never materialized; retain only the policy, not checkpoint weights.
    record = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if not isinstance(record, dict) or set(record) != {'metadata', 'model'}:
        raise ValueError('Context policy requires a complete raw-text checkpoint record')
    metadata, state = record['metadata'], record['model']
    if not isinstance(metadata, dict) or not isinstance(state, dict):
        raise ValueError('Context policy requires dictionary metadata and model state')
    policy = metadata.get('context_policy', 'tc_prefix_context')
    if policy not in ('tc_prefix_context', 'music_only_global_bias'):
        raise ValueError('Unsupported deployment context_policy')
    if policy == 'music_only_global_bias' and (type(metadata.get('format_version')) is not int
            or metadata['format_version'] != 3 or metadata.get('state_scope') != 'music_to_h3_text'
            or 'adapter.global_context_bias' not in state
            or any('context_projection' in name for name in state)):
        raise ValueError('Music-only deployment requires explicit V3 global-bias metadata and no context projection')
    del metadata, state, record
    return policy


def _validate_inference_input(data, context_policy):
    keys = {'waveform'} if context_policy == 'music_only_global_bias' else {'waveform', 'context'}
    if not isinstance(data, dict) or set(data) != keys:
        raise ValueError('Deployment input keys must match checkpoint context_policy exactly; no text/cache/length inputs')
    waveform = data['waveform']
    context = data.get('context')
    if not torch.is_tensor(waveform) or waveform.shape != (2, 162000) or not torch.isfinite(waveform).all():
        raise ValueError('Invalid shared-loader waveform')
    if context_policy != 'music_only_global_bias' and (not torch.is_tensor(context)
            or context.shape != (5120,) or not torch.isfinite(context).all()):
        raise ValueError('Legacy deployment requires finite raw5120 context')
    return waveform, context


class MusicTextEncodingProxy:
    def __init__(self, checkpoint, python='/opt/venv_music/bin/python', legacy_clip=False):
        self.legacy_clip = bool(legacy_clip)
        self.checkpoint = str(Path(checkpoint).resolve(strict=True))
        self.checkpoint_sha256 = sha256(self.checkpoint)
        self.python = python
        self.cache_key = None
        self.cache = None
        self.calls = 0
        self._context_policy = None

    @property
    def context_policy(self):
        if self._context_policy is None:
            self._context_policy = 'tc_prefix_context' if self.legacy_clip else _checkpoint_context_policy(self.checkpoint)
        return self._context_policy

    def __call__(self, waveform, context=None):
        if self.context_policy == 'music_only_global_bias' and context is not None:
            raise ValueError('Music-only checkpoint forbids external context, including zero or cached TC context')
        waveform = waveform.detach().cpu().float().contiguous()
        if context is not None:
            context = context.detach().cpu().float().contiguous()
        inputs = dict(waveform=waveform)
        if self.context_policy != 'music_only_global_bias':
            inputs['context'] = context
        _validate_inference_input(inputs, self.context_policy)
        key = hashlib.sha256(self.checkpoint_sha256.encode() + waveform.numpy().tobytes()
                             + (b'' if context is None else context.numpy().tobytes())).hexdigest()
        if key != self.cache_key:
            with tempfile.TemporaryDirectory(prefix='h3_music_text_') as directory:
                input_path, output_path = Path(directory) / 'input.pt', Path(directory) / 'output.pt'
                torch.save(inputs, input_path)
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
                        or data['predicted_length'] != len(tail) or not torch.isfinite(tail).all()
                        or data.get('context_policy', 'tc_prefix_context') != self.context_policy):
                    raise ValueError('Invalid deployed raw text output')
                self.cache = data
                self.cache_key = key
        self.calls += 1
        print('[music text deployment] ' + json.dumps(dict(call=self.calls,
              checkpoint_sha256=self.checkpoint_sha256, input_sha256=key,
              tail_shape=list(self.cache['tail'].shape), predicted_length=self.cache['predicted_length'],
              uses_ground_truth_caption=False, final_adapter_projection_executed=True,
              legacy_clip_complete_chain=self.legacy_clip, context_policy=self.context_policy,
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
            if not no_text_encoder and self.proxy.context_policy == 'music_only_global_bias':
                raise ValueError('Music-only checkpoint requires no_text_encoder deployment, not a TC prefix path')
            self.waveform = None

        def set_waveform(self, waveform):
            if waveform.shape != (2, 162000):
                raise ValueError('Student text path requires the shared fixed-length32k waveform')
            self.waveform = waveform

        def process(self, pipe, prompt, keyframes=None, ref_blocks=None, height=None, width=None, text_embedding=None):
            if no_text_encoder:
                if prompt is not None or text_embedding is not None or keyframes or self.waveform is None:
                    raise ValueError('No-TC student requires only the bound music waveform and no text inputs')
                music_only = self.proxy.context_policy == 'music_only_global_bias'
                # Only legacy checkpoints receive a zero vector to accommodate their old signature.
                tail = self.proxy(self.waveform, None if music_only else torch.zeros(5120))
                return _no_tc_tail_output(pipe, prompt, keyframes, text_embedding, tail,
                    dict(kind='music_only_global_bias_no_tc_deployment' if music_only else 'legacy_checkpoint_no_tc_music_only_diagnostic',
                         context_policy=self.proxy.context_policy,
                         context_source='none_music_only_global_bias' if music_only else 'zeros_legacy_checkpoint_compatibility',
                         old_model_not_trained_for_no_tc=not music_only, uses_ground_truth_caption=False,
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
            pure_text = metadata.get('kind') == 'pure_text_h3_raw_tokens'
            if pure_text:
                from music_text_condition import validate_pure_text_record
                validate_pure_text_record(record, sample_id)
                original = Path(metadata['source_cache_file'])
                if sha256(original) != metadata['source_cache_file_sha256']:
                    raise ValueError('Bound pure-text teacher source cache file changed')
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
            self.report = dict(kind='pure_text_teacher_no_tc_diagnostic' if pure_text else 'cached_teacher_tail_only_no_tc_diagnostic',
                teacher_only_diagnostic=True, music_prediction=False,
                context_source='none_teacher_tail_only_diagnostic', source_sha256=source_sha256,
                source_path=str(path), original_teacher_tokens=len(tail),
                teacher_tail_sha256=digest.hexdigest(), sample_id=sample_id,
                image_path=str(image_path), audio_path=str(audio_path), student_reads_caption=False)
            if pure_text:
                self.report.update(source_kind=metadata['kind'], text_only=True, image_conditioned=False,
                    source_scope='caption_only_no_reference_context', teacher_length_for_diagnostic_only=True,
                    caption_sha256=metadata['caption_sha256'],
                    source_token_ids_sha256=metadata['source_token_ids_sha256'],
                    source_cache_file=metadata['source_cache_file'],
                    source_cache_file_sha256=metadata['source_cache_file_sha256'],
                    reference_image_binding_only=True)

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


def _music_only_none_cpu_audit():
    import contextlib
    import io
    import sys
    from types import SimpleNamespace
    from unittest.mock import patch
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PromptEmbedder

    waveform = torch.arange(2 * 162000, dtype=torch.float32).reshape(2, 162000) / 162000
    tail = torch.arange(2 * 5120, dtype=torch.float32).reshape(2, 5120) / 5120
    worker_inputs, generated, cuda_calls, rejected = [], [], [], []

    class MockWorker:
        def __init__(self, path, expected_backend=None):
            record = torch.load(path, map_location='cpu', weights_only=True)
            self.checkpoint_metadata = record['metadata']
            self.adapter = SimpleNamespace(context_policy=record['metadata'].get('context_policy', 'tc_prefix_context'), qmax=2)

        def cuda(self):
            cuda_calls.append('mock_model_only')
            return self

        def eval(self):
            return self

        def generate(self, wave, context):
            assert wave.shape == (1, 2, 162000)
            if self.adapter.context_policy == 'music_only_global_bias':
                assert context is None
            else:
                assert context.shape == (1, 5120) and torch.equal(context, torch.zeros(1, 5120))
            generated.append(dict(context_policy=self.adapter.context_policy, context_is_none=context is None))
            return [tail.clone()], torch.tensor([2])

    def fake_subprocess(command, **kwargs):
        path = Path(command[command.index('--input') + 1])
        inputs = torch.load(path, map_location='cpu', weights_only=True)
        worker_inputs.append(sorted(inputs))
        with patch.object(sys, 'argv', ['infer'] + command[2:]), contextlib.redirect_stdout(io.StringIO()):
            main()
        return subprocess.CompletedProcess(command, 0, '', '')

    class StrictPipe:
        def __init__(self):
            self.device = 'cpu'
            self.text_encoder = self.processor = self.tokenizer = None

        def __getattribute__(self, name):
            if name in ('text_encoder', 'processor', 'tokenizer', 'load_models_to_device'):
                raise AssertionError('No-TC accessed text processing: ' + name)
            return object.__getattribute__(self, name)

    def forbidden(*args, **kwargs):
        raise AssertionError('No-TC invoked original text processing')

    with tempfile.TemporaryDirectory(prefix='h3_music_only_none_cpu_') as directory:
        parent = Path(directory)
        paths = {}
        for policy in ('music_only_global_bias', 'tc_prefix_context'):
            path = parent / (policy + '.pt')
            metadata = dict(format_version=3 if policy == 'music_only_global_bias' else 2,
                            state_scope='music_to_h3_text')
            state = {'adapter.context_projection.bias': torch.zeros(8)}
            if policy == 'music_only_global_bias':
                metadata['context_policy'] = policy
                state = {'adapter.global_context_bias': torch.zeros(8)}
            torch.save(dict(metadata=metadata, model=state), path)
            paths[policy] = path
        logs = io.StringIO()
        with patch(__name__ + '.MusicToH3Text', MockWorker), patch(__name__ + '.load_music_text_checkpoint', MockWorker), patch.object(
                torch.Tensor, 'cuda', lambda value, *args, **kwargs: value), patch.object(subprocess, 'run', fake_subprocess), contextlib.redirect_stdout(logs):
            new = MusicTextEncodingProxy(paths['music_only_global_bias'])
            with patch(__name__ + '._checkpoint_context_policy', wraps=_checkpoint_context_policy) as policy_reader:
                first = new(waveform)
                second = new(waveform, None)
                assert policy_reader.call_count == 1
            assert torch.equal(first, tail) and torch.equal(second, tail)
            new_expected_key = hashlib.sha256(new.checkpoint_sha256.encode() + waveform.numpy().tobytes()).hexdigest()
            assert new.cache_key == new_expected_key and new.calls == 2
            for context in (torch.zeros(5120), torch.ones(5120), 'cached_TC_sentinel'):
                try:
                    new(waveform, context)
                except ValueError:
                    rejected.append('new_external_context:' + type(context).__name__)
                else:
                    raise AssertionError('New proxy accepted external context')
            old = MusicTextEncodingProxy(paths['tc_prefix_context'])
            old_tail = old(waveform, torch.zeros(5120))
            old_expected_key = hashlib.sha256(old.checkpoint_sha256.encode() + waveform.numpy().tobytes() + torch.zeros(5120).numpy().tobytes()).hexdigest()
            assert old.cache_key == old_expected_key and torch.equal(old_tail, tail)
            try:
                old(waveform)
            except ValueError:
                rejected.append('legacy_proxy_none')
            else:
                raise AssertionError('Legacy proxy accepted missing context')
            assert worker_inputs == [['waveform'], ['context', 'waveform']]
            assert generated == [dict(context_policy='music_only_global_bias', context_is_none=True),
                                 dict(context_policy='tc_prefix_context', context_is_none=False)]
            for index, (policy, inputs) in enumerate([
                    ('music_only_global_bias', dict(waveform=waveform, context=torch.zeros(5120))),
                    ('music_only_global_bias', dict(waveform=waveform, context=None)),
                    ('tc_prefix_context', dict(waveform=waveform)),
                    ('music_only_global_bias', dict(waveform=waveform, caption='forbidden'))]):
                input_path, output_path = parent / ('bad_' + str(index) + '.pt'), parent / ('out_' + str(index) + '.pt')
                torch.save(inputs, input_path)
                checkpoint = paths[policy]
                argv = ['infer', '--checkpoint', str(checkpoint), '--checkpoint_sha256', sha256(checkpoint),
                        '--input', str(input_path), '--out', str(output_path)]
                before = len(cuda_calls)
                with patch.object(sys, 'argv', argv):
                    try:
                        main()
                    except ValueError:
                        rejected.append('worker_keys:' + policy + ':' + ','.join(sorted(inputs)))
                    else:
                        raise AssertionError('Worker accepted policy-incompatible PT keys')
                assert len(cuda_calls) == before and not output_path.exists()
            with patch.object(MiniMaxH3Unit_PromptEmbedder, 'process', forbidden), patch.object(
                    MiniMaxH3Unit_PromptEmbedder, 'preprocess_ref_blocks', forbidden):
                embedder = student_text_embedder(paths['music_only_global_bias'], no_text_encoder=True)
                assert embedder.onload_model_names == ()
                embedder.set_waveform(waveform)
                output = embedder.process(StrictPipe(), None)
                assert torch.equal(output['prompt_embeds'], tail.bfloat16())
                assert torch.equal(output['text_token_tags'], torch.ones(2, dtype=torch.long))
                try:
                    student_text_embedder(paths['music_only_global_bias'], no_text_encoder=False)
                except ValueError:
                    rejected.append('new_checkpoint_TC_factory')
                else:
                    raise AssertionError('New checkpoint was accepted into TC prefix deployment')
        records = [json.loads(line[len('[no tc condition] '):]) for line in logs.getvalue().splitlines()
                   if line.startswith('[no tc condition] ')]
        assert len(records) == 1 and records[0]['old_model_not_trained_for_no_tc'] is False
        assert records[0]['context_source'] == 'none_music_only_global_bias'
        assert records[0]['context_policy'] == 'music_only_global_bias'
        assert records[0]['semantic_text'] is None and records[0]['prefix_length'] == 0
        invalid = parent / 'invalid_new_state.pt'
        torch.save(dict(metadata=dict(format_version=3, state_scope='music_to_h3_text', context_policy='music_only_global_bias'),
                        model={'adapter.context_projection.weight': torch.zeros(8, 5120)}), invalid)
        try:
            _checkpoint_context_policy(invalid)
        except ValueError:
            rejected.append('new_metadata_context_projection')
        else:
            raise AssertionError('New metadata accepted legacy context projection state')
    return dict(complete=True, cpu_only=True, real_checkpoint_or_gpu_executed=False,
        fake_checkpoint_worker_executed_in_process=True, new_input_keys_exact_waveform_only=True,
        new_generate_context_none=True, new_key_has_no_context_bytes=True,
        legacy_key_input_and_raw_tail_unchanged=True, legacy_input_keys_exact_waveform_context=True,
        checkpoint_policy_read_once_mmap_and_no_state_retained=True,
        new_no_tc_factory_none_bf16_tail_and_no_prefix_exact=True,
        new_metadata_context_projection_rejected=True, no_tc_condition=records[0],
        rejected_inputs=rejected, worker_input_keys=worker_inputs, mock_generated=generated)


def _no_tc_manager_cpu_audit():
    import contextlib
    import copy
    import io
    import sys
    from unittest.mock import patch
    import music_text_oracle_generate as manager

    condition = dict(kind='legacy_checkpoint_no_tc_music_only_diagnostic',
        loaded=False, calls=0, semantic_text=None, processor_loaded=False,
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
        torch.save(dict(metadata=dict(format_version=2), model={}), checkpoint)
        root = Path(__file__).resolve().parents[3]
        scenarios = [(False, None, None, None), (True, None, None, None)] + mutations
        for index, (teacher, target, field, value) in enumerate(scenarios):
            records = copy.deepcopy(dict(condition=condition, native=native, deployment=deployment))
            if teacher:
                records['condition'].update(kind='cached_teacher_tail_only_no_tc_diagnostic',
                    teacher_only_diagnostic=True, music_prediction=False,
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


def _pure_text_teacher_cpu_audit():
    import contextlib
    import copy
    import csv
    import io
    import sys
    from unittest.mock import patch
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PromptEmbedder
    from music_text_condition import (bind_pure_text_cases, validate_pure_text_record,
                                      tensor_sha256, PURE_TEXT_TEACHER_KIND)
    import music_text_oracle_generate as manager

    class ForbiddenPipe:
        def __init__(self):
            self.device = 'cpu'
            self.text_encoder = self.processor = self.tokenizer = None
            self.accesses = 0

        def __getattribute__(self, name):
            if name in ('text_encoder', 'processor', 'tokenizer', 'load_models_to_device'):
                state = object.__getattribute__(self, '__dict__')
                state['accesses'] += 1
                raise AssertionError('Pure teacher accessed TC processing: ' + name)
            return object.__getattribute__(self, name)

    def forbidden(*args, **kwargs):
        raise AssertionError('Pure teacher invoked image/text processing or CUDA')

    rejected, records, commands = [], [], []
    with tempfile.TemporaryDirectory(prefix='h3_pure_teacher_cpu_') as directory:
        parent = Path(directory)
        source_dir, exp_root = parent / 'source', parent / 'exp'
        source_dir.mkdir()
        exp_root.mkdir()
        (exp_root / 'E2').mkdir()
        csv_path = exp_root / 'E2/full_joint.csv'
        cases = [dict(sample_id='cpu_' + str(index), audio_path='/nonexistent/cpu_' + str(index) + '.wav',
                      image_path='/nonexistent/cpu_' + str(index) + '.png') for index in range(3)]
        with csv_path.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['sample_id', 'image_path', 'audio_path'])
            writer.writeheader()
            writer.writerows(cases)
        source = dict(kind=PURE_TEXT_TEACHER_KIND, text_only=True, image_conditioned=False, prefix_length=0,
                      lengths={}, token_ids_sha256={})
        for index, case in enumerate(cases):
            sid = case['sample_id']
            ids = torch.arange(index + 2, dtype=torch.long)
            source['lengths'][sid] = len(ids)
            source['token_ids_sha256'][sid] = tensor_sha256(ids)
        (source_dir / 'plan.json').write_text(json.dumps(source))
        protocol_sha = sha256(source_dir / 'plan.json')
        rows = []
        for index, case in enumerate(cases):
            sid, length = case['sample_id'], index + 2
            tail = (torch.arange(length * 5120, dtype=torch.float32).reshape(length, 5120) / 1024).bfloat16()
            ids = torch.arange(length, dtype=torch.long)
            prompt = 'The dancer moves to music ' + str(index)
            metadata = dict(sample_id=sid, kind=PURE_TEXT_TEACHER_KIND, text_only=True, image_conditioned=False,
                            prefix_length=0, tail_length=length, teacher_length=length, prompt=prompt,
                            caption_sha256=hashlib.sha256(prompt.encode()).hexdigest(), tail_sha256=tensor_sha256(tail),
                            source_token_ids_sha256=tensor_sha256(ids), token_ids_sha256=tensor_sha256(ids),
                            source_protocol_path=str(source_dir / 'plan.json'), source_protocol_sha256=protocol_sha)
            record = dict(metadata=metadata, tail=tail, source_token_ids=ids, teacher_length=length)
            path = source_dir / (sid + '.pt')
            torch.save(record, path)
            rows.append(dict(sample_id=sid, path=str(path), file_sha256=sha256(path), teacher_length=length))
            records.append(record)
        summary = dict(complete=True, expected=3, rows=rows, source=source, source_protocol_sha256=protocol_sha)
        (source_dir / 'finished.json').write_text(json.dumps(summary))
        mutations = [
            ('metadata', 'text_only', 1), ('metadata', 'image_conditioned', 0),
            ('metadata', 'prefix_length', False), ('metadata', 'prefix_length', 1),
            ('metadata', 'kind', 'oracle_teacher_hidden_not_music_prediction'),
            ('metadata', 'caption_sha256', '0' * 64), ('metadata', 'tail_sha256', '0' * 64),
            ('metadata', 'source_token_ids_sha256', '0' * 64), ('metadata', 'token_ids_sha256', '0' * 64),
            ('metadata', 'tail_length', 2.0), ('metadata', 'teacher_length', True),
            ('metadata', 'prefix_context', torch.zeros(5120)),
            ('record', 'prefix_context', torch.zeros(5120)), ('record', 'teacher_length', 2.0),
            ('record', 'tail', records[0]['tail'].float()),
            ('record', 'source_token_ids', torch.arange(3)),
            ('record', 'source_token_ids', torch.arange(2).float())]
        for target, key, value in mutations:
            bad = copy.deepcopy(records[0])
            (bad['metadata'] if target == 'metadata' else bad)[key] = value
            try:
                validate_pure_text_record(bad, cases[0]['sample_id'])
            except ValueError:
                rejected.append(target + '.' + key + ':' + type(value).__name__)
            else:
                raise AssertionError('Pure teacher accepted mutated schema: ' + key)
        binding = parent / 'binding'
        logs = io.StringIO()
        pipe = ForbiddenPipe()
        with patch.object(MiniMaxH3Unit_PromptEmbedder, 'process', forbidden), patch.object(
                MiniMaxH3Unit_PromptEmbedder, 'preprocess_ref_blocks', forbidden), patch(
                'PIL.Image.open', forbidden), patch.object(torch.cuda, '_lazy_init', forbidden), contextlib.redirect_stdout(logs):
            report = bind_pure_text_cases(source_dir, exp_root, binding)
            declared = manager.teacher_tail_cache_source(binding / 'cache')
            assert declared['kind'] == PURE_TEXT_TEACHER_KIND
            assert declared['source_scope'] == 'caption_only_no_reference_context'
            teacher = cached_teacher_tail_embedder(binding / 'cache')
            assert teacher.onload_model_names == ()
            for index, case in enumerate(cases):
                teacher.set_case(case['sample_id'], case['image_path'], case['audio_path'])
                output = teacher.process(pipe, None, ref_blocks=[object()])
                assert torch.equal(output['prompt_embeds'], records[index]['tail'])
                assert torch.equal(output['text_token_tags'], torch.ones(index + 2, dtype=torch.long))
            for kwargs in [dict(prompt=' '), dict(prompt=None, keyframes=[object()]),
                           dict(prompt=None, text_embedding=torch.zeros(1, 5120))]:
                try:
                    teacher.process(pipe, **kwargs)
                except ValueError:
                    rejected.append('no_tc_input:' + ','.join(kwargs))
                else:
                    raise AssertionError('Pure teacher accepted semantic/keyframe/cached text')
            try:
                teacher.set_case(cases[0]['sample_id'], '/wrong.png', cases[0]['audio_path'])
            except ValueError:
                rejected.append('current_case_image_path')
            else:
                raise AssertionError('Pure teacher accepted a different current case image')
            changed = copy.deepcopy(summary)
            changed['source']['token_ids_sha256'][cases[0]['sample_id']] = '0' * 64
            (source_dir / 'finished.json').write_text(json.dumps(changed))
            try:
                bind_pure_text_cases(source_dir, exp_root, parent / 'bad_binding')
            except ValueError:
                rejected.append('manifest_protocol_integrity')
            else:
                raise AssertionError('Binding accepted changed source protocol')
            (source_dir / 'finished.json').write_text(json.dumps(summary))
        no_tc_records = [json.loads(line[len('[no tc condition] '):]) for line in logs.getvalue().splitlines()
                         if line.startswith('[no tc condition] ')]
        assert len(no_tc_records) == 3 and pipe.accesses == 0
        root = Path(__file__).resolve().parents[3]
        native_base = dict(text_encoder_loaded=False, tokenizer_loaded=False, processor_loaded=False,
            text_encoder_calls=0, semantic_text=None, prefix_tokens=0, visual_reference_rows=6,
            visual_reference_sha256='a' * 64, native_music_bypass=False, all_target_audio_clean=True,
            waveform_sha256='b' * 64, reference_sha256='c' * 64, retake_sha256='d' * 64, mask_sha256='e' * 64)
        for mutate in (False, True):
            out = parent / ('manager_bad' if mutate else 'manager_ok')
            argv = ['manager', '--root', str(root), '--out', str(out), '--hosts', 'cpu_mock_host',
                    '--joint_only', '--no_text_encoder', '--teacher_tail_cache', str(binding / 'cache')]
            def fake_run(command, **kwargs):
                if kwargs.get('capture_output'):
                    return subprocess.CompletedProcess(command, 0, 'CPU_MOCK_ADMISSION', '')
                commands.append(command)
                shard = int(__import__('shlex').split(command[-1])[__import__('shlex').split(command[-1]).index('--shard') + 1])
                condition = copy.deepcopy(no_tc_records[shard])
                if mutate:
                    condition['source_scope'] = 'reference_context'
                native = dict(native_base, prompt_embeds_shape=condition['tail_shape'])
                kwargs['stdout'].write('[no tc condition] ' + json.dumps(condition) + '\n')
                kwargs['stdout'].write('[no tc native isolation] ' + json.dumps(native) + '\n[h3ar] done\n')
                return subprocess.CompletedProcess(command, 0)
            with patch.object(sys, 'argv', argv), patch.object(manager.subprocess, 'run', fake_run), contextlib.redirect_stdout(io.StringIO()):
                try:
                    manager.main()
                except ValueError:
                    if not mutate:
                        raise
                    rejected.append('manager_caption_only_scope')
                else:
                    if mutate:
                        raise AssertionError('Manager accepted non-caption-only evidence')
                    plan = json.loads((out / 'plan.json').read_text())
                    assert plan['kind'] == 'pure_text_teacher_no_tc_diagnostic'
                    assert plan['source_scope'] == 'caption_only_no_reference_context'
                    assert plan['semantic_student_prompt'] is None and plan['prefix_length'] == 0
                    assert json.loads((out / 'jobs.json').read_text())['all_sampler_jobs_exit_zero'] is True
        legacy = parent / 'looks_like_pure_text'
        legacy.mkdir()
        assert manager.teacher_tail_cache_source(legacy)['kind'] == 'cached_image_conditioned_teacher'
        original = source_dir / (cases[0]['sample_id'] + '.pt')
        original.write_bytes(original.read_bytes() + b'changed')
        try:
            teacher.set_case(cases[0]['sample_id'], cases[0]['image_path'], cases[0]['audio_path'])
        except ValueError:
            rejected.append('source_cache_file_sha')
        else:
            raise AssertionError('Pure teacher accepted a changed source cache')
    return dict(complete=True, cpu_only=True, gpu_started=False, actual_lm_forward=False, image_io=False,
                pure_text_schema_hash_token_ids_length_checked=True, fixed_joint_three_case_binding_checked=True,
                source_manifest_and_protocol_checked=True, current_case_path_and_source_sha_checked=True,
                teacher_tail_only_bf16_and_one_tags_exact=True, prefix_length=0,
                text_encoder_tokenizer_processor_accesses=pipe.accesses,
                original_prompt_process_and_preprocess_calls=0, manager_uses_manifest_not_directory_name=True,
                manager_three_joint_no_tc_commands_checked=True, no_tc_conditions=no_tc_records,
                rejected_inputs=rejected, scope='CPU fixtures only; no teacher LM, image/VAE/DiT/GPU/video execution or quality claim')


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
        context_policy = 'tc_prefix_context'

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
        pure_text_teacher_audit=_pure_text_teacher_cpu_audit(),
        music_only_none_deployment_audit=_music_only_none_cpu_audit(),
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
    if not isinstance(data, dict) or set(data) not in ({'waveform'}, {'waveform', 'context'}):
        raise ValueError('Inference only accepts exact music-only or legacy context inputs; never caption/length/cache')
    if args.legacy_clip:
        waveform, context = _validate_inference_input(data, 'tc_prefix_context')
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
        context_policy = model.adapter.context_policy
        waveform, context = _validate_inference_input(data, context_policy)
        model = model.cuda().eval()
        with torch.no_grad():
            tails, lengths = model.generate(waveform.cuda()[None], None if context is None else context.cuda()[None])
        result = dict(tail=tails[0].cpu(), predicted_length=int(lengths[0]),
                      query_max=model.adapter.qmax, checkpoint_sha256=args.checkpoint_sha256,
                      metadata=model.checkpoint_metadata, uses_ground_truth_caption=False,
                      context_policy=context_policy)
    with args.out.open('xb') as stream:
        torch.save(result, stream)
    print(json.dumps(dict(event='deployed_raw_text', tokens=result['predicted_length'],
                          query_max=result['query_max'], checkpoint_sha256=args.checkpoint_sha256,
                          legacy_clip_complete_chain=args.legacy_clip,
                          context_policy=result.get('context_policy', 'tc_prefix_context'))), flush=True)


if __name__ == '__main__':
    main()
