"""Frozen-H3 prediction supervision with an exact detached-tail VJP boundary.

The music environment owns trainable parameters; the H3 environment returns the
loss derivative with respect to the identical FP32 student tail. No inference
proxy, teacher length, GT velocity target or trainable H3 parameter is used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save_new(path, value, binary=False):
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    with path.open('xb' if binary else 'x') as stream:
        if binary:
            torch.save(value, stream)
        else:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    result = hashlib.sha256(str(value.dtype).encode() + str(tuple(value.shape)).encode())
    result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def prepare(args):
    sys.path.insert(0, str(ROOT / 'DiffSynth-Studio-upstream-20260930'))
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
    from diffsynth.models.minimax_h3_dit import pack_audio
    ms = args.exp_root / 'MS'
    split = json.loads((ms / 'music_scale_v4_train1800_split.json').read_text())
    ids = split['train'][:3] + split['val'][:1]
    index = dict((sid, path) for path, sid in (
        line.split('\t') for line in args.cache_list.read_text().splitlines() if line))
    if any(sid not in index for sid in ids):
        raise ValueError('Selected IDs are missing from the original high-resolution cache')
    pipe = MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[ModelConfig(path=str(args.weights / 'audio_vae/model.safetensors'))], processor_config=None)
    pipe.audio_vae.requires_grad_(False).eval()
    silence = torch.zeros(2, 162000, device='cuda', dtype=torch.float32)
    with torch.no_grad():
        reference_audio = pipe.audio_vae.encode_audio(silence, dtype=torch.bfloat16)
    records = []
    for sid in ids:
        prefix = torch.load(ms / 'music_text_online_context_v1' / (sid + '.pt'), map_location='cpu', weights_only=True)
        shared, positive, _ = torch.load(index[sid], map_location='cpu', weights_only=False)
        if (not torch.equal(prefix['prefix'], positive['prompt_embeds'][:len(prefix['prefix'])])
                or not torch.equal(prefix['context'], prefix['prefix'][-1])):
            raise ValueError('Existing reference latent cache does not match the actual empty text prefix')
        target = None
        for shard in range(4):
            directory = ms / f'music_text_targets_neutral_v1_s{shard}'
            finished = json.loads((directory / 'finished.json').read_text())
            if finished.get('complete') is not True:
                raise ValueError('Teacher cache is incomplete')
            for row in finished['rows']:
                if row['sample_id'] == sid:
                    if target is not None or digest(row['path']) != row['file_sha256']:
                        raise ValueError('Teacher target missing identity or duplicated')
                    target = torch.load(row['path'], map_location='cpu', weights_only=True)
        if target is None or not torch.equal(target['prefix_context'], prefix['context']):
            raise ValueError('Teacher structural context mismatch')
        visual = shared['ref_visual_anchor']
        width, height = target['metadata']['prepared_image_size']
        if visual.shape != ((width // 32) * (height // 32), 96):
            raise ValueError('Reference rows differ from declared prepared image')
        audio_t = shared['audio_input_latents'].shape[-1]
        if reference_audio.shape[-1] > audio_t:
            raise ValueError('Fixed silence reference exceeds target audio duration')
        retake = torch.cat([reference_audio, reference_audio[..., -1:].repeat(1, 1, audio_t - reference_audio.shape[-1])], dim=-1)
        records.append(dict(sample_id=sid, partition='train' if sid in split['train'] else 'val',
            prefix=prefix['prefix'], prefix_tags=prefix['prefix_tags'], context=prefix['context'],
            teacher_tail=target['tail'], target_video_latents=shared['input_latents'],
            ref_visual_anchor=visual, ref_audio_anchor=pack_audio(reference_audio).cpu(),
            input_latents_audio=retake.cpu(), denoise_mask_audio=torch.zeros_like(retake).cpu(),
            ref_blocks=[dict(kind='image', latent_t=1, latent_h=height // 16, latent_w=width // 16, ref_audio_t=0),
                        dict(kind='audio', ref_audio_t=reference_audio.shape[-1])],
            metadata=dict(cache_path=index[sid], cache_sha256=digest(index[sid]),
                image_sha256=target['metadata']['image_sha256'], teacher_caption_sha256=target['metadata']['caption_sha256'],
                prepared_image_size=[width, height], prefix_exact=True, native_music_bypass=False,
                silence_waveform_sha256=tensor_digest(silence), ref_audio_sha256=tensor_digest(pack_audio(reference_audio)),
                retake_sha256=tensor_digest(retake), mask_sha256=tensor_digest(torch.zeros_like(retake)))))
    result = dict(records=records, metadata=dict(split_sha256=digest(ms / 'music_scale_v4_train1800_split.json'),
        selected_train=3, selected_val=1, frozen_preprocessing_only=True, student_reads_teacher_caption=False,
        fixed_native_silence=True, cache_shape=[1, 24, 37, 80, 44], teacher_prediction_loss_not_gt_velocity=True))
    save_new(args.out, result, binary=True)
    save_new(args.out.with_suffix('.json'), dict(complete=True, cache_sha256=digest(args.out),
        metadata=result['metadata'], rows=[dict(sample_id=r['sample_id'], partition=r['partition'],
                                             metadata=r['metadata']) for r in records]))
    print(json.dumps(dict(event='behavior_cache_finished', records=len(records), out=str(args.out))), flush=True)


def export_music(args):
    from music_text_adapter import load_music_text_checkpoint
    model = load_music_text_checkpoint(args.checkpoint, expected_backend='musicfm').cuda().eval()
    bundle = torch.load(args.bundle, map_location='cpu', weights_only=True)
    record = torch.load(args.cache, map_location='cpu', weights_only=True)['records'][args.index]
    sid = record['sample_id']
    wave, context = bundle['waveforms'][sid].cuda()[None], record['context'].cuda().float()[None]
    with torch.no_grad():
        condition = model.student_condition(wave, context)
    tail = condition['tails'][0].cpu()
    result = dict(sample_id=sid, tail=tail, predicted_length=int(condition['lengths'][0]),
        waveform=bundle['waveforms'][sid], context=record['context'],
        checkpoint_sha256=digest(args.checkpoint), checkpoint=str(args.checkpoint),
        bundle_sha256=digest(args.bundle), cache_sha256=digest(args.cache), tail_sha256=tensor_digest(tail),
        student_condition='unchanged full-qmax and own predicted length', teacher_length_input=False)
    save_new(args.out, result, binary=True)
    print(json.dumps(dict(event='behavior_student_export', tokens=len(tail), checkpoint_sha256=result['checkpoint_sha256'])), flush=True)


def entry_export(args):
    from music_generation_validate import fixed_cases
    from music_clip_bridge import load_audio
    from music_text_adapter import MusicToH3Text, load_music_text_checkpoint
    cases = fixed_cases(args.exp_root)
    joint = [case for case in cases if case['condition'] == 'joint']
    swapped = {case['sample_id']: case for case in cases if case['condition'] == 'aswap'}
    if len(joint) != 3 or set(swapped) != {case['sample_id'] for case in joint}:
        raise ValueError('Require exactly the same three fixed review references')
    checkpoint_sha = digest(args.checkpoint)
    model = load_music_text_checkpoint(args.checkpoint, expected_backend='musicfm').cuda().eval()
    if (not isinstance(model, MusicToH3Text) or not model.online_training
            or model.freeze_backbone or model.checkpoint_metadata['format_version'] != 2):
        raise ValueError('Require the complete dense-trained online MusicFM plus full adapter checkpoint')
    model.requires_grad_(False)
    before = model_hashes(model)
    records = []
    with torch.no_grad():
        for case in joint:
            source = args.oracle_cache / (Path(case['filename']).stem + '.pt')
            oracle = torch.load(source, map_location='cpu', weights_only=True)
            meta = oracle['metadata']
            sid = case['sample_id']
            context = oracle['prefix_context']
            lp = len(oracle['prefix_tags'])
            if (meta['kind'] != 'oracle_teacher_hidden_not_music_prediction'
                    or meta['case']['sample_id'] != sid or meta['case']['condition'] != 'joint'
                    or meta['case']['audio_path'] != case['audio_path']
                    or meta['prefix_length'] != lp or meta['tail_length'] != len(oracle['tail'])
                    or context.shape != (5120,) or oracle['tail'].shape[1] != 5120
                    or tensor_digest(oracle['tail']) != meta['teacher_tail_sha256']
                    or not torch.equal(oracle['token_ids'], torch.tensor(meta['teacher_ids'], dtype=torch.long))):
                raise ValueError('Fixed-case oracle identity or structural context mismatch')
            variants = {}
            for name, audio_case in (('original', case), ('aswap', swapped[sid]), ('zero', case)):
                audio_path = Path(audio_case['audio_path'])
                waveform, effective_frames = load_audio(audio_path.stem, audio_root=audio_path.parent)
                if name == 'zero':
                    waveform = torch.zeros_like(waveform)
                # Only this current-image structural context and shared-loader audio reach the student.
                condition = model.student_condition(waveform.cuda()[None], context.cuda().float()[None])
                raw = condition['raw_tokens'][0].cpu()
                tail = condition['tails'][0].cpu()
                length = int(condition['lengths'][0])
                if (raw.shape != (model.adapter.qmax, 5120) or length != len(tail)
                        or not torch.equal(raw[:length], tail)):
                    raise ValueError('Full-qmax output and own predicted-length truncation mismatch')
                variants[name] = dict(raw_tokens=raw, tail=tail, waveform=waveform,
                    log_length=condition['log_length'][0].cpu(), predicted_length=length,
                    audio_id=audio_path.stem, audio_path=str(audio_path), audio_file_sha256=digest(audio_path),
                    effective_frames=effective_frames, waveform_sha256=tensor_digest(waveform),
                    raw_tokens_sha256=tensor_digest(raw), tail_sha256=tensor_digest(tail),
                    zero_input=name == 'zero', teacher_length_input=False, tc_reencoding=False)
            records.append(dict(sample_id=sid, case=case, context=context, context_sha256=tensor_digest(context),
                prefix_ids=oracle['token_ids'][:lp], prefix_tags=oracle['prefix_tags'],
                teacher_tail=oracle['tail'], teacher_length=len(oracle['tail']),
                teacher_tail_sha256=tensor_digest(oracle['tail']), oracle_metadata=meta,
                oracle_source=str(source), oracle_sha256=digest(source),
                image_path=meta['case']['image_path'], image_sha256=digest(meta['case']['image_path']),
                variants=variants))
            print(json.dumps(dict(event='entry_music_case_exported', sample_id=sid,
                lengths={name: value['predicted_length'] for name, value in variants.items()})), flush=True)
    after = model_hashes(model)
    if (before != after or any(p.grad is not None or p.requires_grad for p in model.parameters())
            or digest(args.checkpoint) != checkpoint_sha):
        raise ValueError('Read-only entry export changed music state/checkpoint or received gradients')
    sources = [Path(__file__), Path(sys.modules['music_text_adapter'].__file__),
               Path(sys.modules['music_clip_bridge'].__file__), Path(sys.modules['music_encoder_adapter'].__file__)]
    result = dict(format_version=1, kind='real_music_full_tail_entry_export', complete=True,
        checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_sha, query_max=model.adapter.qmax,
        records=records, optimizer_updates=0, teacher_caption_or_length_student_input=False,
        tc_loaded=False, waveform_loader='music_clip_bridge.load_audio',
        historical_cached_tc_context_interface=True, new_noTC_student_interface=False,
        context_source='same-reference oracle cached TC prefix last row; historical student only',
        fixed_case_csv_sha256={condition: digest(args.exp_root / 'E2' / ('full_' + condition + '.csv'))
            for condition in ('joint', 'aswap')},
        source_sha256={str(path): digest(path) for path in sources},
        music_state_before_sha256=before, music_state_after_sha256=after,
        music_state_unchanged=True, quality_claim=False)
    save_new(args.out, result, binary=True)
    save_new(args.out.with_suffix('.json'), {key: value for key, value in result.items() if key != 'records'})
    print(json.dumps(dict(event='entry_export_finished', out=str(args.out), cases=3, optimizer_updates=0)), flush=True)


def entry_export_no_tc(args):
    from music_generation_validate import fixed_cases
    from music_clip_bridge import load_audio
    from music_text_adapter import MusicToH3Text, load_music_text_checkpoint
    cases = fixed_cases(args.exp_root)
    joint = [case for case in cases if case['condition'] == 'joint']
    swapped = {case['sample_id']: case for case in cases if case['condition'] == 'aswap'}
    if len(joint) != 3 or set(swapped) != {case['sample_id'] for case in joint}:
        raise ValueError('Require exactly three fixed review cases and their swapped audio')
    checkpoint_sha = digest(args.checkpoint)
    model = load_music_text_checkpoint(args.checkpoint, expected_backend='musicfm').cuda().eval()
    if (not isinstance(model, MusicToH3Text) or not model.online_training
            or model.freeze_backbone or model.checkpoint_metadata['format_version'] != 2):
        raise ValueError('Require complete online dense-trained MusicFM plus full adapter')
    model.requires_grad_(False)
    before = model_hashes(model)
    # Existing C300 needs a context argument, but no TC data supplies it in this new diagnostic.
    context = torch.zeros(5120, dtype=torch.float32)
    records = []
    with torch.no_grad():
        for case in joint:
            sid = case['sample_id']
            variants = {}
            for name, audio_case in (('original', case), ('aswap', swapped[sid]), ('zero', case)):
                audio_path = Path(audio_case['audio_path'])
                waveform, effective_frames = load_audio(audio_path.stem, audio_root=audio_path.parent)
                if name == 'zero':
                    waveform = torch.zeros_like(waveform)
                condition = model.student_condition(waveform.cuda()[None], context.cuda()[None])
                raw, tail = condition['raw_tokens'][0].cpu(), condition['tails'][0].cpu()
                length = int(condition['lengths'][0])
                if (raw.shape != (model.adapter.qmax, 5120) or length != len(tail)
                        or not torch.equal(raw[:length], tail)):
                    raise ValueError('NoTC full-qmax and self-length boundary mismatch')
                variants[name] = dict(raw_tokens=raw, tail=tail, waveform=waveform,
                    log_length=condition['log_length'][0].cpu(), predicted_length=length,
                    audio_id=audio_path.stem, audio_path=str(audio_path), audio_file_sha256=digest(audio_path),
                    effective_frames=effective_frames, waveform_sha256=tensor_digest(waveform),
                    raw_tokens_sha256=tensor_digest(raw), tail_sha256=tensor_digest(tail),
                    zero_input=name == 'zero', teacher_length_input=False,
                    context_source='explicit FP32 zeros5120, no TC or cached context')
            # Teacher cache is read only AFTER the three student forwards, for comparison evidence only.
            source = args.oracle_cache / (Path(case['filename']).stem + '.pt')
            oracle = torch.load(source, map_location='cpu', weights_only=True)
            meta, teacher = oracle['metadata'], oracle['tail']
            if (meta['kind'] != 'oracle_teacher_hidden_not_music_prediction'
                    or meta['case']['sample_id'] != sid or meta['case']['condition'] != 'joint'
                    or meta['case']['audio_path'] != case['audio_path']
                    or teacher.ndim != 2 or teacher.shape[-1] != 5120 or teacher.dtype != torch.bfloat16
                    or meta['tail_length'] != len(teacher)
                    or tensor_digest(teacher) != meta['teacher_tail_sha256']):
                raise ValueError('NoTC diagnostic teacher tail/source identity mismatch')
            records.append(dict(sample_id=sid, case=case, teacher_tail=teacher,
                teacher_length=len(teacher), teacher_tail_sha256=tensor_digest(teacher),
                oracle_source=str(source), oracle_sha256=digest(source),
                teacher_compare_only=True, variants=variants))
            print(json.dumps(dict(event='entry_no_tc_music_case_exported', sample_id=sid,
                lengths={name: value['predicted_length'] for name, value in variants.items()})), flush=True)
    after = model_hashes(model)
    if (before != after or digest(args.checkpoint) != checkpoint_sha
            or any(p.requires_grad or p.grad is not None for p in model.parameters())):
        raise ValueError('NoTC export changed music state/checkpoint or received gradients')
    sources = [Path(__file__), Path(sys.modules['music_text_adapter'].__file__),
        Path(sys.modules['music_clip_bridge'].__file__), Path(sys.modules['music_encoder_adapter'].__file__)]
    result = dict(format_version=1, kind='real_music_no_tc_entry_export', complete=True,
        checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_sha, query_max=model.adapter.qmax,
        records=records, zero_context=context, zero_context_sha256=tensor_digest(context),
        context_source='explicit FP32 zeros5120', prefix_present=False,
        old_model_not_trained_for_no_tc=True, textencoder_loaded=False, textencoder_calls=0,
        teacher_caption_or_length_student_input=False, cached_tc_context_student_input=False,
        optimizer_updates=0, gradients_absent=True, quality_claim=False,
        waveform_loader='music_clip_bridge.load_audio',
        fixed_case_csv_sha256={condition: digest(args.exp_root / 'E2' / ('full_' + condition + '.csv'))
            for condition in ('joint', 'aswap')},
        source_sha256={str(path): digest(path) for path in sources},
        music_state_before_sha256=before, music_state_after_sha256=after, music_state_unchanged=True,
        scope='Unchanged complete C300 with zero context and full own-length music tails; distribution-shift diagnostic, not a noTC-trained model or visual-conditioned student.')
    save_new(args.out, result, binary=True)
    save_new(args.out.with_suffix('.json'), {key: value for key, value in result.items()
        if key not in ('records', 'zero_context')})
    print(json.dumps(dict(event='entry_export_no_tc_finished', out=str(args.out), cases=3, textencoder_calls=0)), flush=True)


def entry_compare_no_tc(args):
    # No pipeline, processor, prompt embedder, prefix or music proxy is constructed in this stage.
    sys.path.insert(0, str(ROOT / 'DiffSynth-Studio-upstream-20260930'))
    export_sha = digest(args.input)
    export = torch.load(args.input, map_location='cpu', weights_only=True)
    if (export.get('kind') != 'real_music_no_tc_entry_export' or export.get('complete') is not True
            or len(export.get('records', [])) != 3 or export.get('prefix_present') is not False
            or export.get('textencoder_loaded') is not False or export.get('textencoder_calls') != 0
            or export.get('cached_tc_context_student_input') is not False
            or export.get('teacher_caption_or_length_student_input') is not False
            or export.get('old_model_not_trained_for_no_tc') is not True
            or export.get('optimizer_updates') != 0
            or digest(export['checkpoint']) != export['checkpoint_sha256']
            or export['zero_context'].shape != (5120,) or torch.count_nonzero(export['zero_context'])
            or tensor_digest(export['zero_context']) != export['zero_context_sha256']
            or export['music_state_before_sha256'] != export['music_state_after_sha256']):
        raise ValueError('Require an unchanged complete real noTC zero-context export, not historical cached-context data')
    modules, native = entry_load_native(args)
    before = model_hashes(modules)
    rows = []
    with torch.no_grad():
        for record in export['records']:
            if (digest(record['oracle_source']) != record['oracle_sha256']
                    or tensor_digest(record['teacher_tail']) != record['teacher_tail_sha256']
                    or record['teacher_tail'].dtype != torch.bfloat16
                    or record['teacher_length'] != len(record['teacher_tail'])):
                raise ValueError('NoTC teacher comparison source/tail changed')
            tails = {'teacher_H': record['teacher_tail']}
            for name, variant in record['variants'].items():
                if (name not in ('original', 'aswap', 'zero')
                        or tensor_digest(variant['raw_tokens']) != variant['raw_tokens_sha256']
                        or tensor_digest(variant['tail']) != variant['tail_sha256']
                        or tensor_digest(variant['waveform']) != variant['waveform_sha256']
                        or variant['raw_tokens'].shape != (export['query_max'], 5120)
                        or variant['predicted_length'] != len(variant['tail'])
                        or not torch.equal(variant['raw_tokens'][:variant['predicted_length']], variant['tail'])):
                    raise ValueError('NoTC real student export boundary changed')
                tails[name] = variant['tail']
            if set(tails) != {'teacher_H', 'original', 'aswap', 'zero'}:
                raise ValueError('NoTC compare requires all three real music input variants')
            lengths = {name: len(value) for name, value in tails.items()}
            visible = min(lengths.values())
            outputs, statistics, layouts, quantization = {}, {}, {}, {}
            for name, tail in tails.items():
                if tail.ndim != 2 or tail.shape[-1] != 5120 or not len(tail) or not torch.isfinite(tail).all():
                    raise ValueError('NoTC entry requires finite nonempty full tail tokens')
                length = len(tail)
                native_raw = tail.cuda().to(torch.bfloat16)
                cu = torch.tensor([0, length, length], dtype=torch.int32, device='cuda')
                projected = modules['condition_proj'](native_raw)
                refined = modules['token_refiner'](projected, cu_seqlens=cu, max_seqlen=length)
                outputs[name] = dict(raw5120=tail.cpu(), raw5120_h3_bf16=native_raw.cpu(),
                    condition_proj5376=projected.cpu(), token_refiner5376=refined.cpu())
                layouts[name] = dict(natural_tail_length=length, refiner_cu_seqlens=cu.cpu().tolist(),
                    refiner_max_seqlen=length, prefix_rows=0, full_H3_packed_layout_evaluated=False)
                statistics[name] = {stage: dict(natural_tail=entry_tensor_stats(value),
                    common_visible_tail=entry_tensor_stats(value[:visible]), tensor_sha256=tensor_digest(value))
                    for stage, value in outputs[name].items()}
                quantization[name] = entry_row_difference(tail, native_raw.cpu())
                del native_raw, projected, refined
            pairs = []
            names = list(tails)
            for i, first in enumerate(names):
                for second in names[i + 1:]:
                    overlap = min(lengths[first], lengths[second])
                    pairs.append(dict(first=first, second=second,
                        natural_lengths=[lengths[first], lengths[second]], common_visible_tail_rows=visible,
                        natural_pair_overlap_rows=overlap, same_length=lengths[first] == lengths[second],
                        full_packed_exact_claim=False, common_interval_is_not_equal_layout_intervention=True,
                        stages={stage: dict(common_teacher_student_visible_tail=entry_row_difference(
                            outputs[first][stage][:visible], outputs[second][stage][:visible]),
                            natural_pair_overlap_tail=entry_row_difference(outputs[first][stage][:overlap], outputs[second][stage][:overlap]))
                            for stage in outputs[first]}))
            rows.append(dict(sample_id=record['sample_id'], natural_tail_lengths=lengths,
                common_visible_teacher_tail_rows=visible, prefix_rows=0, layouts=layouts,
                statistics=statistics, pairs=pairs, raw_to_h3_bf16_quantization=quantization,
                full_qmax_raw_student_statistics={name: entry_tensor_stats(value['raw_tokens'])
                    for name, value in record['variants'].items()},
                teacher_compare_only=True, old_model_not_trained_for_no_tc=True))
            print(json.dumps(dict(event='entry_no_tc_case_compared', sample_id=record['sample_id'],
                natural_lengths=lengths, common_rows=visible)), flush=True)
    after = model_hashes(modules)
    if (before != after or digest(args.input) != export_sha
            or digest(export['checkpoint']) != export['checkpoint_sha256']
            or any(p.requires_grad or p.grad is not None for p in modules.parameters())):
        raise ValueError('Frozen noTC entry modules/export/checkpoint changed or received gradients')
    report = dict(complete=True, kind='native_H3_no_tc_real_music_vs_teacher',
        export_source=str(args.input), export_sha256=export_sha, checkpoint_sha256=export['checkpoint_sha256'],
        textencoder_loaded=False, textencoder_calls=0, prefix_rows=0, processor_loaded=False,
        music_proxy_reencoding=False, video_vae_loaded=False, audio_vae_loaded=False,
        old_model_not_trained_for_no_tc=True, full_H3_packed_layout_evaluated=False,
        teacher_tail_and_length_compare_only=True, optimizer_updates=0, gradients_absent=True,
        independent_native_entry=native, native_entry_before_sha256=before, native_entry_after_sha256=after,
        all_module_states_unchanged=True, cases=rows, quality_claim=False,
        behavior_source_sha256=digest(Path(__file__)),
        native_module_source_sha256=digest(sys.modules[modules['token_refiner'].__class__.__module__].__file__),
        attention_implementation=os.environ.get('DIFFSYNTH_ATTENTION_IMPLEMENTATION'),
        torch_version=str(torch.__version__), native_entry_dtype='torch.bfloat16',
        scope='Tail-only real C300 zero-context raw5120, native condition_proj5376 and full two-layer token_refiner, own [0,L,L]/max L. No TC/prefix/full packed/VAE/video/updates; common-position metrics are not equal-layout interventions or quality acceptance.')
    save_new(args.out, report, binary=True)
    save_new(args.out.with_suffix('.json'), report)
    print(json.dumps(dict(event='entry_compare_no_tc_finished', out=str(args.out), cases=3,
        textencoder_calls=0, decoder_blocks_loaded=0)), flush=True)


def entry_tensor_stats(value):
    value = value.detach().float().cpu()
    if value.ndim != 2 or not len(value) or not torch.isfinite(value).all():
        raise ValueError('Entry statistics require finite nonempty token rows')
    norms = value.norm(dim=-1)
    result = dict(shape=list(value.shape), mean=float(value.mean()), std=float(value.std(unbiased=False)),
        rms=float(value.square().mean().sqrt()), min=float(value.min()), max=float(value.max()),
        row_norm_min=float(norms.min()), row_norm_max=float(norms.max()), row_norm_mean=float(norms.mean()),
        row_mean_min=float(value.mean(-1).min()), row_mean_max=float(value.mean(-1).max()),
        row_rms_min=float(value.square().mean(-1).sqrt().min()), row_rms_max=float(value.square().mean(-1).sqrt().max()))
    if len(value) > 1:
        row_delta = value[1:] - value[:-1]
        cosine = torch.nn.functional.cosine_similarity(value[1:], value[:-1], dim=-1)
        result.update(adjacent_row_difference_rms=float(row_delta.square().mean().sqrt()),
            adjacent_row_difference_norm_mean=float(row_delta.norm(dim=-1).mean()),
            adjacent_row_cosine_mean=float(cosine.mean()), adjacent_row_cosine_min=float(cosine.min()),
            centered_row_rms=float((value - value.mean(0)).square().mean().sqrt()))
    return result


def entry_row_difference(first, second):
    if first.shape != second.shape or first.ndim != 2 or not len(first):
        raise ValueError('Common-position comparison requires identical nonempty intervals')
    a, b = first.float().cpu(), second.float().cpu()
    delta = a - b
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return dict(compared_rows=len(a), same_interval_exact=torch.equal(first, second),
        mse=float(delta.square().mean()), rms=float(delta.square().mean().sqrt()),
        max_abs=float(delta.abs().max()), relative_rms=float(delta.norm() / a.norm().clamp_min(1e-12)),
        cosine_mean=float(cosine.mean()), cosine_min=float(cosine.min()), cosine_max=float(cosine.max()),
        row_difference_norm_mean=float(delta.norm(dim=-1).mean()),
        row_difference_norm_max=float(delta.norm(dim=-1).max()))


def entry_load_native(args):
    from safetensors import safe_open
    from diffsynth.models.minimax_h3_dit import MiniMaxH3TokenRefiner
    from diffsynth.utils.state_dict_converters.minimax_h3_dit import MiniMaxH3DiTSingularityStateDictConverter
    directory = args.weights / 'transformer'
    config_path = directory / 'config.json'
    config = json.loads(config_path.read_text())
    expected = dict(hidden_size=5376, text_dim=5120, token_refiner_num_layers=2,
        num_attention_heads=56, attention_head_dim=128, ffn_hidden_size=14336,
        norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Unsupported native entry configuration; use the real full H3 pipeline, never guessed weights')
    # These are the exact module constructors used by MiniMaxH3DiT, with no decoder blocks allocated.
    with torch.device('meta'):
        modules = torch.nn.ModuleDict(dict(
            condition_proj=torch.nn.Linear(config['text_dim'], config['hidden_size'], bias=True),
            token_refiner=MiniMaxH3TokenRefiner(config['token_refiner_num_layers'], config['hidden_size'],
                config['num_attention_heads'], config['attention_head_dim'], config['ffn_hidden_size'],
                config['norm_eps'], config['qk_norm_eps'], config['final_norm_eps'])))
    wanted = modules.state_dict()
    state, provenance = {}, {}
    shards = sorted(directory.glob('model*.safetensors'))
    if len(shards) != 13:
        raise ValueError('Require the original 13-shard Ref2VA native source')
    for path in shards:
        with safe_open(str(path), framework='pt', device='cpu') as stream:
            for key in stream.keys():
                mapped = key
                if key.startswith('model.diffusion_model.'):
                    converted = MiniMaxH3DiTSingularityStateDictConverter({key: None})
                    mapped, = converted
                if mapped not in wanted:
                    if mapped.startswith(('condition_proj.', 'token_refiner.')):
                        raise ValueError('Unrecognized native entry weight: ' + key)
                    continue
                if mapped in state:
                    raise ValueError('Duplicate mapped native entry weight: ' + mapped)
                tensor = stream.get_tensor(key)
                if tensor.shape != wanted[mapped].shape or tensor.dtype != torch.bfloat16:
                    raise ValueError('Native entry source shape/dtype mismatch: ' + key)
                state[mapped] = tensor
                provenance[mapped] = dict(source_key=key, safetensors=str(path), shape=list(tensor.shape),
                    dtype=str(tensor.dtype), tensor_sha256=tensor_digest(tensor), source_file_bytes=path.stat().st_size)
    if set(state) != set(wanted):
        raise ValueError('Missing real native entry weights: ' + ', '.join(sorted(set(wanted) - set(state))))
    modules.load_state_dict(state, strict=True, assign=True)
    modules.to(device='cuda', dtype=torch.bfloat16).requires_grad_(False).eval()
    if model_hashes(modules) != {name: row['tensor_sha256'] for name, row in provenance.items()}:
        raise ValueError('Loaded native entry modules differ from their exact source tensors')
    return modules, dict(config=config, config_source=str(config_path), config_sha256=digest(config_path),
        mapped_weights=provenance, decoder_blocks_loaded=0, independently_loaded_real_native_modules=True,
        converter_source_sha256=digest(Path(sys.modules[MiniMaxH3DiTSingularityStateDictConverter.__module__].__file__)))


def entry_compare(args):
    if not args.legacy_tc_prefix_diagnostic:
        raise ValueError('Legacy TC-prefix diagnostic disabled under noTC student protocol; explicit legacy-only flag required')
    from PIL import Image
    sys.path.insert(0, str(ROOT / 'DiffSynth-Studio-upstream-20260930'))
    from diffsynth.pipelines.minimax_h3_audio_video import (
        MiniMaxH3Pipeline, ModelConfig, MiniMaxH3Unit_PromptEmbedder,
        MiniMaxH3Unit_ReferenceEncoder, MiniMaxH3Unit_PackedSequenceBuilder)
    from music_text_condition import prefix_length, replace_tail
    from music_text_infer import MusicTextEncodingProxy
    export_sha = digest(args.input)
    export = torch.load(args.input, map_location='cpu', weights_only=True)
    if (export.get('complete') is not True or export.get('kind') != 'real_music_full_tail_entry_export'
            or len(export.get('records', [])) != 3 or export.get('optimizer_updates') != 0
            or export.get('teacher_caption_or_length_student_input') is not False
            or digest(export['checkpoint']) != export['checkpoint_sha256']):
        raise ValueError('Require a complete unchanged real three-case music export')
    text_shards = sorted((args.weights / 'text_encoder').glob('model*.safetensors'))
    if not text_shards:
        raise FileNotFoundError(args.weights / 'text_encoder')
    vram = dict(offload_dtype=torch.bfloat16, offload_device='cpu', onload_dtype=torch.bfloat16,
        onload_device='cuda', preparing_dtype=torch.bfloat16, preparing_device='cuda',
        computation_dtype=torch.bfloat16, computation_device='cuda')
    pipe = MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[ModelConfig(path=[str(path) for path in text_shards], **vram)],
        processor_config=ModelConfig(path=str(args.weights / 'processor')))
    pipe.text_encoder.requires_grad_(False).eval()
    tc_before = model_hashes(pipe.text_encoder)
    prompt_unit, ref_unit, builder = (MiniMaxH3Unit_PromptEmbedder(),
        MiniMaxH3Unit_ReferenceEncoder(), MiniMaxH3Unit_PackedSequenceBuilder())
    prefixes, recipes = {}, {}
    # Recover the original sampler recipe, not a caller-selected/default prepared size.
    sampler = ROOT / 'H3-posttrain/scripts_omnidance/exp/h3ar/sample_h3_ar.py'
    import ast
    tree = ast.parse(sampler.read_text())
    generate, = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'generate']
    sh, = [node.value for node in generate.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'sh' for target in node.targets)]
    recipe = {ast.literal_eval(key): value for key, value in zip(sh.keys, sh.values)}
    short_edge = ast.literal_eval(recipe['ref_image_short_edge'])
    num_frames = ast.literal_eval(recipe['num_frames'])
    if short_edge != 2048 or num_frames != 124:
        raise ValueError('Original oracle reference recipe changed; require its actual prepared reference source')
    with torch.no_grad():
        for record in export['records']:
            sid = record['sample_id']
            if (digest(record['oracle_source']) != record['oracle_sha256']
                    or digest(record['image_path']) != record['image_sha256']
                    or tensor_digest(record['context']) != record['context_sha256']
                    or tensor_digest(record['teacher_tail']) != record['teacher_tail_sha256']):
                raise ValueError('Exported oracle/reference tensors or sources changed')
            final_oracle_path = Path(record['oracle_source']).with_suffix('.json')
            final_oracle = json.loads(final_oracle_path.read_text())
            if (final_oracle.get('status') != 'passed' or final_oracle.get('packed_checked') is not True
                    or final_oracle.get('teacher_hidden_sha256') != record['oracle_metadata']['teacher_hidden_sha256']):
                raise ValueError('Source oracle final same-H review audit did not pass')
            source_directory = Path(record['oracle_source']).parent.parent
            integrity_path = source_directory / 'joint_integrity.json'
            integrity = json.loads(integrity_path.read_text())
            source_row, = [row for row in integrity['rows'] if row['case']['sample_id'] == sid]
            video_stream, = [stream for stream in source_row['integrity']['streams'] if stream['codec_type'] == 'video']
            if (integrity.get('complete') is not True or source_row['case'] != record['case']
                    or video_stream['decoded_frames'] != num_frames or video_stream['fps'] != 24.):
                raise ValueError('Original review target dimensions/protocol are not verified')
            target_shape = [1, 24, ((num_frames - 5) // 17) * 5 + 2,
                video_stream['height'] // 16, video_stream['width'] // 16]
            target_audio_shape = [2, 32, round(num_frames / 24. * 40.)]
            log_path = source_directory / 'logs' / ('shard_' + str(record['case']['shard']) + '.log')
            import re
            log_text = log_path.read_text()
            inputs, = [json.loads(line.split('[h3ar] input_conditions ', 1)[1]) for line in log_text.splitlines()
                       if line.startswith('[h3ar] input_conditions ')]
            padded, = re.findall(r'\[h3ar\] repeat_clean audio tail: (\d+) rows fixed at effective time=1', log_text)
            ref_audio_t = target_audio_shape[-1] - int(padded)
            if (inputs['image_path'] != record['image_path'] or inputs['audio_path'] != record['case']['audio_path']
                    or inputs['native_audio_condition'] != 'fixed_silence' or inputs['text_hidden_oracle'] is not True
                    or inputs['semantic_text'] != ' ' or ref_audio_t < 1):
                raise ValueError('Source oracle actual reference route changed')
            with Image.open(record['image_path']) as image:
                width, height = ref_unit._resolve_reference_image_shape(pipe, *image.size, short_edge)
                prepared = image.convert('RGB').resize((width, height), Image.LANCZOS)
            ref_blocks = [dict(kind='image', prepared_image=prepared, latent_t=1,
                latent_h=height // 16, latent_w=width // 16, ref_audio_t=0),
                dict(kind='audio', ref_audio_t=ref_audio_t)]
            ids, tags = prompt_unit.preprocess_ref_blocks(pipe, ' ', ref_blocks)[:2]
            lp = prefix_length(pipe.tokenizer, ' ', ids)
            if (lp != len(record['prefix_tags']) or not torch.equal(ids[:lp].cpu(), record['prefix_ids'])
                    or not torch.equal(tags[:lp].cpu(), record['prefix_tags'])):
                raise ValueError('Actual source prepared reference IDs/tags do not match the same-case oracle')
            encoded = prompt_unit.process(pipe, ' ', ref_blocks=ref_blocks)
            prefix = encoded['prompt_embeds'][:lp].cpu().clone()
            if not torch.equal(prefix[-1], record['context']):
                raise ValueError('Actual empty-prefix TC context is not exact against the same-case oracle')
            if not torch.equal(encoded['text_token_tags'][:lp].cpu(), record['prefix_tags']):
                raise ValueError('Actual TC prefix tags changed')
            teacher_condition = replace_tail(prefix, record['prefix_tags'], record['teacher_tail'])
            same_h = tensor_digest(teacher_condition['prompt_embeds']) == record['oracle_metadata']['teacher_hidden_sha256']
            if not same_h:
                raise ValueError('Source oracle same-H replacement full raw hidden hash is not exact')
            prefixes[sid] = prefix
            recipes[sid] = dict(prepared_image_size=[width, height], prepared_image_sha256=
                hashlib.sha256(prepared.tobytes()).hexdigest(), ref_image_short_edge=short_edge,
                reference_recipe_source=str(sampler), reference_recipe_source_sha256=digest(sampler),
                recipe_reconstructed_not_cached=True, recipe_verified_by_same_oracle_ids_tags_context=True,
                ref_blocks=[{key: value for key, value in block.items() if key != 'prepared_image'} for block in ref_blocks],
                target_video_shape=target_shape, target_audio_shape=target_audio_shape,
                target_source_integrity_sha256=digest(integrity_path), source_log_sha256=digest(log_path),
                source_final_oracle_report_sha256=digest(final_oracle_path),
                source_ref_audio_length_from_target_minus_logged_padding=True,
                source_oracle_same_H_replacement_exact=True, prefix_rows=lp, prefix_sha256=tensor_digest(prefix))
            del encoded, teacher_condition
    pipe.load_models_to_device([])
    tc_after = model_hashes(pipe.text_encoder)
    if tc_before != tc_after or any(p.requires_grad or p.grad is not None for p in pipe.text_encoder.parameters()):
        raise ValueError('Frozen TC changed state or received gradients')
    # The actual existing deployment proxy replays the exact complete student with the reconstructed context.
    proxy = MusicTextEncodingProxy(export['checkpoint'], python=args.music_python)
    with torch.no_grad():
        for record in export['records']:
            for name, variant in record['variants'].items():
                if (tensor_digest(variant['waveform']) != variant['waveform_sha256']
                        or tensor_digest(variant['raw_tokens']) != variant['raw_tokens_sha256']
                        or tensor_digest(variant['tail']) != variant['tail_sha256']
                        or len(variant['tail']) != variant['predicted_length']
                        or not torch.equal(variant['raw_tokens'][:variant['predicted_length']], variant['tail'])):
                    raise ValueError('Exported complete student boundary changed')
                replay = proxy(variant['waveform'], prefixes[record['sample_id']][-1])
                if not torch.equal(replay, variant['tail']):
                    raise ValueError('Real music student and existing deployment proxy same-tail replay is not exact: ' + name)
    modules, native = entry_load_native(args)
    before = model_hashes(modules)
    rows = []
    layout_pipe = pipe
    # Empty placeholders carry only source target dimensions; no noise, prediction or trajectory is evaluated.
    with torch.no_grad():
        for record in export['records']:
            sid, prefix, tags = record['sample_id'], prefixes[record['sample_id']], record['prefix_tags']
            lp = len(prefix)
            tails = dict(teacher_H=record['teacher_tail'], **{name: value['tail'] for name, value in record['variants'].items()})
            lengths = {name: len(tail) for name, tail in tails.items()}
            visible = min(lengths.values())
            ref_blocks = recipes[sid]['ref_blocks']
            video = torch.empty(recipes[sid]['target_video_shape'], device='meta')
            audio = torch.empty(recipes[sid]['target_audio_shape'], device='meta')
            outputs, layouts, statistics = {}, {}, {}
            for name, tail in tails.items():
                condition = replace_tail(prefix, tags, tail)
                packed = builder.process(layout_pipe, video_latents=video, audio_latents=audio,
                    ref_blocks=ref_blocks, **condition)['packed']
                text_len = len(packed['text_pos'])
                # Identical to native model_fn_minimax_h3, using each condition's own packed text_len.
                cu = torch.tensor([0, text_len, text_len], dtype=torch.int32, device='cuda')
                raw = condition['prompt_embeds'].cuda().to(torch.bfloat16)
                projected = modules['condition_proj'](raw)
                refined = modules['token_refiner'](projected, cu_seqlens=cu, max_seqlen=text_len)
                outputs[name] = dict(raw5120=torch.cat((prefix.float(), tail.float())),
                    raw5120_h3_bf16=raw.cpu(), condition_proj5376=projected.cpu(), token_refiner5376=refined.cpu())
                layouts[name] = dict(natural_tail_length=len(tail), natural_text_length=text_len,
                    packed_seq_len=packed['seq_len'], packed_cu_seqlens=packed['cu_seqlens'].tolist(),
                    refiner_cu_seqlens=cu.cpu().tolist(), refiner_max_seqlen=text_len,
                    packed_tensor_sha256={key: tensor_digest(value) for key, value in packed.items() if torch.is_tensor(value)})
                statistics[name] = {stage: dict(prefix=entry_tensor_stats(value[:lp]),
                    natural_tail=entry_tensor_stats(value[lp:]), common_visible_tail=entry_tensor_stats(value[lp:lp + visible]),
                    tensor_sha256=tensor_digest(value)) for stage, value in outputs[name].items()}
                if name == 'teacher_H':
                    same = replace_tail(condition['prompt_embeds'][:lp], tags, condition['prompt_embeds'][lp:])
                    same_packed = builder.process(layout_pipe, video_latents=video, audio_latents=audio,
                        ref_blocks=ref_blocks, **same)['packed']
                    exact_packed = all(torch.equal(value, same_packed[key]) if torch.is_tensor(value)
                        else value == same_packed[key] for key, value in packed.items())
                    same_projection = modules['condition_proj'](same['prompt_embeds'].cuda())
                    same_refined = modules['token_refiner'](same_projection, cu_seqlens=cu, max_seqlen=text_len)
                    if (not exact_packed or not torch.equal(raw.cpu(), same['prompt_embeds'])
                            or not torch.equal(projected, same_projection) or not torch.equal(refined, same_refined)):
                        raise ValueError('Native source oracle same-H replacement entry parity failed')
                del raw, projected, refined
            pairs = []
            names = list(tails)
            for i, first in enumerate(names):
                for second in names[i + 1:]:
                    overlap = min(lengths[first], lengths[second])
                    pairs.append(dict(first=first, second=second,
                        natural_lengths=[lengths[first], lengths[second]], common_visible_tail_rows=visible,
                        natural_pair_overlap_rows=overlap,
                        same_natural_layout=layouts[first] == layouts[second],
                        common_interval_is_not_exact_layout_claim=True,
                        stages={stage: dict(common_prefix=entry_row_difference(outputs[first][stage][:lp], outputs[second][stage][:lp]),
                            common_teacher_student_visible_tail=entry_row_difference(outputs[first][stage][lp:lp + visible], outputs[second][stage][lp:lp + visible]),
                            natural_pair_overlap_tail=entry_row_difference(outputs[first][stage][lp:lp + overlap], outputs[second][stage][lp:lp + overlap]))
                            for stage in outputs[first]}))
            rows.append(dict(sample_id=sid, reference=recipes[sid], layouts=layouts, statistics=statistics,
                natural_tail_lengths=lengths, common_visible_teacher_tail_rows=visible, pairs=pairs,
                full_qmax_raw_student_statistics={name: entry_tensor_stats(value['raw_tokens']) for name, value in record['variants'].items()},
                same_H_source_raw_projection_refiner_and_packed_exact=True,
                real_student_vs_existing_music_proxy_tail_exact=True,
                raw_to_h3_bf16_quantization={name: entry_row_difference(values['raw5120'][lp:], values['raw5120_h3_bf16'][lp:])
                    for name, values in outputs.items()}))
            print(json.dumps(dict(event='entry_case_compared', sample_id=sid, natural_lengths=lengths, common_rows=visible)), flush=True)
    after = model_hashes(modules)
    if before != after or any(p.requires_grad or p.grad is not None for p in modules.parameters()):
        raise ValueError('Frozen native entry modules changed state or received gradients')
    source_modules = [sys.modules[MiniMaxH3Pipeline.__module__], sys.modules[modules['token_refiner'].__class__.__module__],
                      sys.modules['music_text_condition'], sys.modules['music_text_infer']]
    report = dict(complete=True, kind='native_H3_entry_real_music_vs_teacher', export_source=str(args.input),
        export_sha256=export_sha, checkpoint_sha256=export['checkpoint_sha256'], optimizer_updates=0,
        actual_TC_empty_image_prefix_only=True, music_tail_reencoded_by_TC=False,
        legacy_tc_prefix_diagnostic=True, new_noTC_student_interface=False,
        teacher_tail_and_length_compare_only=True, independent_native_entry=native,
        tc_state_before_sha256=tc_before, tc_state_after_sha256=tc_after,
        native_entry_before_sha256=before, native_entry_after_sha256=after,
        all_module_states_unchanged=True, cases=rows, quality_claim=False,
        source_sha256={str(Path(module.__file__)): digest(module.__file__) for module in source_modules},
        behavior_source_sha256=digest(Path(__file__)),
        attention_implementation=os.environ.get('DIFFSYNTH_ATTENTION_IMPLEMENTATION'),
        torch_version=str(torch.__version__), native_entry_dtype='torch.bfloat16',
        scope='Raw5120, real native condition_proj5376 and complete two-layer token_refiner only. Natural own-length layouts retained; common-position statistics are not equal-layout interventions or video quality evidence.')
    save_new(args.out, report, binary=True)
    save_new(args.out.with_suffix('.json'), report)
    print(json.dumps(dict(event='entry_compare_finished', out=str(args.out), cases=3, decoder_blocks_loaded=0)), flush=True)


def model_hashes(model):
    return {name: tensor_digest(value) for name, value in model.state_dict().items()}


def h3_vjp(args):
    sys.path.insert(0, str(ROOT / 'DiffSynth-Studio-upstream-20260930'))
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig, MiniMaxH3Unit_PackedSequenceBuilder
    from music_text_condition import replace_tail
    export = torch.load(args.input, map_location='cpu', weights_only=True)
    record = torch.load(args.cache, map_location='cpu', weights_only=True)['records'][args.index]
    if (export['cache_sha256'] != digest(args.cache) or record['sample_id'] != export['sample_id']
            or not torch.equal(export['context'], record['context'])
            or tensor_digest(export['tail']) != export['tail_sha256']
            or export['predicted_length'] != len(export['tail'])):
        raise ValueError('Student boundary identity mismatch')
    vram = dict(offload_dtype=torch.bfloat16, offload_device='cuda', onload_dtype=torch.bfloat16,
                onload_device='cuda', preparing_dtype=torch.bfloat16, preparing_device='cuda',
                computation_dtype=torch.bfloat16, computation_device='cuda')
    shards = sorted((args.weights / 'transformer').glob('model*.safetensors'))
    if len(shards) != 13:
        raise ValueError('Require the existing complete frozen Ref2VA DiT')
    pipe = MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[ModelConfig(path=[str(p) for p in shards], **vram)], processor_config=None)
    pipe.dit.requires_grad_(False).eval()
    pipe.load_models_to_device(['dit'])
    if args.h3_second_device:
        from types import MethodType
        from diffsynth.core.vram.layers import AutoTorchModule
        second_device = torch.device(args.h3_second_device)
        if second_device.type != 'cuda' or second_device.index == 0:
            raise ValueError('Second H3 device must be a separate visible CUDA GPU')
        for block in pipe.dit.blocks[len(pipe.dit.blocks) // 2:]:
            block.to(second_device)
            for module in block.modules():
                if isinstance(module, AutoTorchModule):
                    module.offload_device = str(second_device)
                    module.onload_device = str(second_device)
                    module.preparing_device = str(second_device)
                    module.computation_device = str(second_device)
            original_forward = block.forward
            def on_second_device(self, value, original_forward=original_forward, **kwargs):
                source_device = value.device
                moved = {key: item.to(second_device) if torch.is_tensor(item) else item
                         for key, item in kwargs.items()}
                return original_forward(value.to(second_device), **moved).to(source_device)
            block.forward = MethodType(on_second_device, block)
    mlp_parity = []
    if args.mlp_chunk:
        from types import MethodType
        from diffsynth.models.minimax_h3_dit import MiniMaxH3MLP
        for module in pipe.dit.modules():
            if not isinstance(module, MiniMaxH3MLP):
                continue
            original = module.forward
            def chunked(self, value, original=original):
                return torch.cat([original(part) for part in value.split(args.mlp_chunk, dim=0)], dim=0)
            if not mlp_parity:
                probe = torch.randn(2 * args.mlp_chunk + 1, module.fc1.in_features,
                                    device='cuda', dtype=torch.bfloat16, requires_grad=True)
                with torch.autograd.graph.save_on_cpu(pin_memory=True):
                    full = original(probe)
                    full_gradient, = torch.autograd.grad(full.float().square().mean(), probe)
                    split = chunked(module, probe)
                    split_gradient, = torch.autograd.grad(split.float().square().mean(), probe)
                output_error = float((full.float() - split.float()).abs().max())
                gradient_error = float((full_gradient.float() - split_gradient.float()).abs().max())
                if output_error > 0.02 or gradient_error > 1e-5:
                    raise ValueError(f'Chunked pointwise MLP parity failed: output={output_error}, gradient={gradient_error}')
                mlp_parity.append(dict(output_max_abs=output_error, gradient_max_abs=gradient_error,
                                       probe_shape=list(probe.shape), chunk_tokens=args.mlp_chunk))
                del probe, full, split, full_gradient, split_gradient
            module.forward = MethodType(chunked, module)
    before = model_hashes(pipe.dit)
    z = record['target_video_latents'][:, :, :args.latent_frames].cuda().to(torch.bfloat16)
    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    noise = torch.randn(z.shape, generator=generator).cuda().to(torch.bfloat16)
    x = ((1 - args.sigma) * z.float() + args.sigma * noise.float()).to(torch.bfloat16)
    common = dict(video_latents=x, audio_latents=record['input_latents_audio'].cuda().to(torch.bfloat16),
        input_latents_audio=record['input_latents_audio'].cuda().to(torch.bfloat16),
        denoise_mask_audio=record['denoise_mask_audio'][:, :1].cuda().to(torch.bfloat16),
        ref_visual_anchor=record['ref_visual_anchor'].cuda().to(torch.bfloat16),
        ref_audio_anchor=record['ref_audio_anchor'].cuda().to(torch.bfloat16),
        imgvid_cond_noise_aug=0.999, audio_cond_noise_aug=1.,
        timestep_video=torch.tensor([args.sigma * 1000], device='cuda'), timestep_audio=torch.tensor([1000.], device='cuda'))
    prefix, tags = record['prefix'].cuda(), record['prefix_tags'].cuda()
    builder = MiniMaxH3Unit_PackedSequenceBuilder()

    def predict(tail, grad):
        condition = replace_tail(prefix, tags, tail)
        packed = builder.process(pipe, video_latents=x, audio_latents=common['audio_latents'],
                                 ref_blocks=record['ref_blocks'], **condition)['packed']
        with torch.set_grad_enabled(grad):
            video, audio = pipe.model_fn(dit=pipe.dit, packed=packed, prompt_embeds=condition['prompt_embeds'],
                use_gradient_checkpointing=grad, use_gradient_checkpointing_offload=grad, **common)
        return video, audio, packed

    started = time.monotonic()
    with torch.no_grad():
        teacher, _, teacher_packed = predict(record['teacher_tail'].cuda(), False)
    leaf = export['tail'].cuda().float().requires_grad_(True)
    torch.cuda.reset_peak_memory_stats()
    if args.h3_second_device:
        torch.cuda.reset_peak_memory_stats(args.h3_second_device)
    with torch.autograd.graph.save_on_cpu(pin_memory=True):
        student, _, student_packed = predict(leaf, True)
        loss = (student.float() - teacher.detach().float()).square().mean()
        grad, = torch.autograd.grad(loss, leaf)
    if not student.requires_grad or not torch.isfinite(grad).all() or float(grad.norm()) <= 0:
        raise ValueError('Frozen H3 did not return a finite nonzero condition gradient')
    if any(p.requires_grad or p.grad is not None for p in pipe.dit.parameters()) or before != model_hashes(pipe.dit):
        raise ValueError('Frozen DiT parameters/buffers changed or received parameter gradients')
    result = dict(sample_id=record['sample_id'], tail_gradient=grad.cpu(), loss=float(loss.detach()),
        original_tail_sha256=export['tail_sha256'], checkpoint_sha256=export['checkpoint_sha256'],
        cache_sha256=export['cache_sha256'], export_sha256=digest(args.input),
        teacher_prediction_sha256=tensor_digest(teacher), student_prediction_sha256=tensor_digest(student),
        teacher_length=len(record['teacher_tail']), student_length=len(leaf),
        h3_state_hashes_unchanged=True, h3_parameter_gradients_absent=True, student_condition_gradient_norm=float(grad.norm()),
        teacher_and_student_use_same_noisy_latents=True, teacher_no_grad=True,
        own_packed_sequence_lengths=[teacher_packed['seq_len'], student_packed['seq_len']],
        sigma=args.sigma, noise_seed=args.seed, latent_frames=args.latent_frames,
        native_music_bypass=False, frozen_cache_native_audio_metadata=record['metadata'],
        actual_native_audio_sha256={key: tensor_digest(common[key]) for key in (
            'ref_audio_anchor', 'input_latents_audio', 'denoise_mask_audio')},
        attention_implementation=os.environ.get('DIFFSYNTH_ATTENTION_IMPLEMENTATION'),
        mlp_chunk_tokens=args.mlp_chunk, pointwise_mlp_parity=mlp_parity,
        h3_second_device=args.h3_second_device,
        cuda_second_peak_bytes=torch.cuda.max_memory_allocated(args.h3_second_device) if args.h3_second_device else 0,
        cuda_peak_bytes=torch.cuda.max_memory_allocated(), prediction_seconds=time.monotonic() - started,
        scope='Frozen teacher/student video velocity consistency and leaf VJP, not GT velocity regression or generation acceptance')
    save_new(args.out, result, binary=True)
    save_new(args.out.with_suffix('.json'), dict(complete=True, **{k: v for k, v in result.items() if k != 'tail_gradient'}))
    print(json.dumps(dict(event='frozen_h3_vjp_finished', loss=result['loss'], grad_norm=result['student_condition_gradient_norm'],
                         peak_bytes=result['cuda_peak_bytes'])), flush=True)


def music_vjp(args):
    from music_text_adapter import load_music_text_checkpoint
    from music_text_contrastive import _state_hashes
    export = torch.load(args.input, map_location='cpu', weights_only=True)
    vjp = torch.load(args.vjp, map_location='cpu', weights_only=True)
    if (digest(args.checkpoint) != export['checkpoint_sha256'] or vjp['checkpoint_sha256'] != export['checkpoint_sha256']
            or vjp['export_sha256'] != digest(args.input) or vjp['original_tail_sha256'] != export['tail_sha256']):
        raise ValueError('VJP checkpoint or exported condition changed')
    model = load_music_text_checkpoint(args.checkpoint, expected_backend='musicfm').cuda()
    model.configure_online_training(True).train()
    before = _state_hashes(model)
    wave, context = export['waveform'].cuda()[None], export['context'].cuda().float()[None]
    condition = model.student_condition(wave, context)
    tail = condition['tails'][0]
    if not torch.equal(tail.detach().cpu(), export['tail']) or len(tail) != export['predicted_length']:
        raise ValueError('VJP replay does not match the complete deployed student condition')
    gradient = vjp['tail_gradient'].cuda()
    if gradient.shape != tail.shape or not torch.isfinite(gradient).all():
        raise ValueError('Invalid H3 condition VJP')
    torch.autograd.backward(tail, gradient)
    groups = dict(backbone=[], adapter=[], length_head=[])
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not parameter.requires_grad or not torch.isfinite(parameter.grad).all():
            raise ValueError('Invalid parameter gradient scope')
        key = 'backbone' if name.startswith('backbone.') else 'length_head' if name.startswith('adapter.length_head.') else 'adapter'
        groups[key].append(dict(name=name, norm=float(parameter.grad.float().norm()), elements=parameter.numel()))
    if not groups['backbone'] or not groups['adapter'] or groups['length_head']:
        raise ValueError('Expected dense/adapter gradient but no gradient through discrete length selection')
    if before != _state_hashes(model):
        raise ValueError('Gradient-only smoke changed parameters or buffers')
    summary = {key: dict(tensors=len(rows), squared_norm_sum=sum(row['norm'] ** 2 for row in rows),
                         nonzero_tensors=sum(row['norm'] > 0 for row in rows), rows=rows) for key, rows in groups.items()}
    if any(summary[key]['nonzero_tensors'] == 0 for key in ('backbone', 'adapter')):
        raise ValueError('H3 behavior loss did not reach the trainable music groups')
    save_new(args.out, dict(complete=True, checkpoint_sha256=export['checkpoint_sha256'],
        exact_student_tail_replay=True, frozen_h3_vjp_report=str(args.vjp.with_suffix('.json')),
        groups=summary, parameters_and_buffers_unchanged=True, optimizer_updates=0,
        length_head_requires_separate_existing_length_loss=True,
        scope='Exact chain-rule VJP replay of frozen-H3 condition loss; no optimizer step or generation claim'))
    print(json.dumps(dict(event='music_vjp_smoke_finished', backbone_tensors=len(groups['backbone']),
                         adapter_tensors=len(groups['adapter']), optimizer_updates=0)), flush=True)


def update_music(args):
    from music_text_adapter import load_music_text_checkpoint, save_music_text_checkpoint, masked_token_loss, length_loss
    from music_text_contrastive import _online_batch, _online_conditions, _online_scores, _state_hashes, _gradient_scope, _evaluate_online, _online_deployment_audit
    if args.behavior_weight not in (0., 1.) or args.index >= 3:
        raise ValueError('Controlled smoke updates only train records and behavior weight0/1')
    export = torch.load(args.input, map_location='cpu', weights_only=True)
    cache = torch.load(args.cache, map_location='cpu', weights_only=True)
    record = cache['records'][args.index]
    bundle = torch.load(args.bundle, map_location='cpu', weights_only=True)
    previous_sha = digest(args.checkpoint)
    if (export['checkpoint_sha256'] != previous_sha or export['cache_sha256'] != digest(args.cache)
            or export['bundle_sha256'] != digest(args.bundle) or record['sample_id'] != export['sample_id']
            or record['partition'] != 'train' or not torch.equal(record['context'], export['context'])
            or not torch.equal(bundle['waveforms'][record['sample_id']], export['waveform'])):
        raise ValueError('Training step does not match its exported complete condition')
    model = load_music_text_checkpoint(args.checkpoint, expected_backend='musicfm').cuda()
    model.configure_online_training(True).train()
    before = _state_hashes(model)
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in model.backbone.parameters() if p.requires_grad], lr=1e-5),
        dict(params=list(model.adapter.parameters()), lr=1e-4)], weight_decay=0.)
    if args.optimizer_state:
        saved = torch.load(args.optimizer_state, map_location='cpu', weights_only=True)
        if saved['checkpoint_sha256'] != previous_sha or saved['behavior_weight'] != args.behavior_weight:
            raise ValueError('Optimizer history belongs to a different checkpoint or experiment')
        optimizer.load_state_dict(saved['optimizer'])
        initial_hashes, history, init_sha = saved['initial_hashes'], saved['history'], saved['init_sha256']
    else:
        initial_hashes, history, init_sha = before, [], previous_sha
    sid = record['sample_id']
    wave, context, target, lengths, mask = _online_batch([sid], bundle, model.adapter.qmax)
    if not torch.equal(context[0].cpu(), export['context'].float()):
        raise ValueError('Training student structural context changed')
    conditions, first, second = _online_conditions(model, [sid], bundle['metadata']['train_ids'], bundle, context, 1)
    tail = conditions[0]['tails'][0]
    if not torch.equal(tail.detach().cpu(), export['tail']) or len(tail) != export['predicted_length']:
        raise ValueError('Exact VJP replay failed; refuse optimizer update')
    behavior_loss = 0.
    if args.behavior_weight:
        vjp = torch.load(args.vjp, map_location='cpu', weights_only=True)
        if (vjp['export_sha256'] != digest(args.input) or vjp['checkpoint_sha256'] != previous_sha
                or vjp['original_tail_sha256'] != export['tail_sha256']
                or vjp['h3_state_hashes_unchanged'] is not True or vjp['latent_frames'] != 37):
            raise ValueError('Full-length frozen-H3 VJP identity or scope failed')
        gradient = vjp['tail_gradient'].cuda()
        if gradient.shape != tail.shape or not torch.isfinite(gradient).all():
            raise ValueError('Nonfinite or mismatched behavior gradient')
        behavior_loss = vjp['loss']
        torch.autograd.backward(tail, gradient * args.behavior_weight, retain_graph=True)
    mse = masked_token_loss(conditions[0]['raw_tokens'], target, mask)['raw_mse']
    ll = length_loss(conditions[0]['log_length'], lengths, model.adapter.qmax)
    mean, std = bundle['mean'].cuda(), bundle['std'].cuda()
    scores = _online_scores(conditions, target, mask, mean, std)
    nce = torch.nn.functional.cross_entropy(scores / 0.07, torch.zeros(1, device='cuda', dtype=torch.long))
    base_loss = mse / bundle['teacher_energy'] + 0.01 * ll + nce
    base_loss.backward()
    gradient_scope = _gradient_scope(model)
    total_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
    optimizer.step()
    after = _state_hashes(model)
    parameters = dict(model.named_parameters())
    changed = [name for name in after if before[name] != after[name]]
    if (not any(name.startswith('backbone.') for name in changed)
            or not any(name.startswith('adapter.') for name in changed)
            or any(name not in parameters or not parameters[name].requires_grad for name in changed)):
        raise ValueError('Incorrect music update or frozen buffer mutation')
    history.append(dict(step=len(history) + 1, sample_id=sid, partition='train',
        raw_mse=float(mse.detach()), length_loss=float(ll.detach()), nce=float(nce.detach()),
        base_loss=float(base_loss.detach()), behavior_loss=behavior_loss,
        combined_loss=float(base_loss.detach()) + args.behavior_weight * behavior_loss,
        student_predicted_length=len(tail), wrong_audio_id=first[0], second_audio_id=second[0],
        exact_student_condition_replay=True, gradient_before_clip_norm=float(total_norm)))
    args.out.mkdir()
    checkpoint = args.out / 'last.pt'
    model.eval()
    save_music_text_checkpoint(checkpoint, model)
    current_sha = digest(checkpoint)
    save_new(args.out / 'optimizer.pt', dict(optimizer=optimizer.state_dict(), checkpoint_sha256=current_sha,
        initial_hashes=initial_hashes, history=history, init_sha256=init_sha, behavior_weight=args.behavior_weight), binary=True)
    changed_total = [name for name in after if after[name] != initial_hashes[name]]
    save_new(args.out / 'weight_update_audit.json', dict(before_sha256=initial_hashes, after_sha256=after,
        actual_changed_parameters=changed_total, backbone_changed_tensors=sum(n.startswith('backbone.') for n in changed_total),
        adapter_changed_tensors=sum(n.startswith('adapter.') for n in changed_total),
        frozen_parameters_and_buffers_unchanged=True, gradient_step=gradient_scope))
    del optimizer, conditions, tail, base_loss, mse, ll, nce, scores
    result = dict(complete=True, checkpoint_sha256=current_sha, steps=len(history), actual_items_seen=len(history),
        actual_backbone_changed_tensors=sum(n.startswith('backbone.') for n in changed_total),
        actual_adapter_changed_tensors=sum(n.startswith('adapter.') for n in changed_total),
        frozen_parameters_and_buffers_unchanged=True, history=history,
        protocol=dict(backbone_train=True, behavior_weight=args.behavior_weight, init_sha256=init_sha,
            bundle_sha256=export['bundle_sha256'], cache_sha256=export['cache_sha256'], batch_size=1,
            length_weight=0.01, clip_weight=1., backbone_lr=1e-5, adapter_lr=1e-4,
            selected_train=[r['sample_id'] for r in cache['records'] if r['partition'] == 'train'],
            heldout_val=[r['sample_id'] for r in cache['records'] if r['partition'] == 'val'],
            student_uses_ground_truth_length=False, source_sha256=digest(Path(__file__)),
            frozen_h3=True, no_gt_fm_velocity_loss=True))
    if args.finalize:
        final = _evaluate_online(model, bundle['metadata']['val_ids'], bundle['metadata']['val_ids'], bundle,
            mean, std, bundle['position_mean'].cuda(), 2)
        save_new(args.out / 'final_validation.json', final)
        w, c, _, _, _ = _online_batch(bundle['metadata']['val_ids'][:1], bundle, model.adapter.qmax)
        result['deployment'] = _online_deployment_audit(model, checkpoint, w, c, args.out)
        result['final'] = final['macro']
    save_new(args.out / 'finished.json', result)
    print(json.dumps(dict(event='behavior_music_update_finished', step=len(history), behavior_weight=args.behavior_weight,
                         checkpoint_sha256=current_sha, final_validation=args.finalize)), flush=True)


def gradient_components(args):
    from music_text_adapter import load_music_text_checkpoint, masked_token_loss, length_loss
    from music_text_contrastive import _online_batch, _online_conditions, _online_scores, _state_hashes
    import math
    export = torch.load(args.input, map_location='cpu', weights_only=True)
    vjp = torch.load(args.vjp, map_location='cpu', weights_only=True)
    bundle = torch.load(args.bundle, map_location='cpu', weights_only=True)
    if (digest(args.checkpoint) != export['checkpoint_sha256'] or digest(args.bundle) != export['bundle_sha256']
            or vjp['checkpoint_sha256'] != export['checkpoint_sha256']
            or vjp['export_sha256'] != digest(args.input) or vjp['original_tail_sha256'] != export['tail_sha256']
            or vjp['h3_state_hashes_unchanged'] is not True or vjp['latent_frames'] != 37
            or export['sample_id'] not in bundle['metadata']['train_ids']):
        raise ValueError('Require the exact existing full-length train VJP and its original checkpoint/input')
    model = load_music_text_checkpoint(args.checkpoint, expected_backend='musicfm').cuda()
    model.configure_online_training(True).train()
    before = _state_hashes(model)
    sid = export['sample_id']
    wave, context, target, lengths, mask = _online_batch([sid], bundle, model.adapter.qmax)
    if not torch.equal(wave[0].cpu(), export['waveform']) or not torch.equal(context[0].cpu(), export['context'].float()):
        raise ValueError('Diagnostic waveform or structural context changed')
    conditions, first, second = _online_conditions(model, [sid], bundle['metadata']['train_ids'], bundle, context, 1)
    tail = conditions[0]['tails'][0]
    if not torch.equal(tail.detach().cpu(), export['tail']):
        raise ValueError('Gradient diagnostic does not reproduce the exported original tail')
    raw = masked_token_loss(conditions[0]['raw_tokens'], target, mask)['raw_mse']
    ll = length_loss(conditions[0]['log_length'], lengths, model.adapter.qmax)
    scores = _online_scores(conditions, target, mask, bundle['mean'].cuda(), bundle['std'].cuda())
    nce = torch.nn.functional.cross_entropy(scores / 0.07, torch.zeros(1, device='cuda', dtype=torch.long))
    tail_gradient = vjp['tail_gradient'].cuda()
    if tail_gradient.shape != tail.shape or not torch.isfinite(tail_gradient).all():
        raise ValueError('Invalid condition VJP')
    losses = {'raw_mse': raw, 'log_length': ll, 'sequence_nce': nce,
              'prediction_vjp': (tail * tail_gradient.detach()).sum()}
    weights = {'raw_mse': 1. / bundle['teacher_energy'], 'log_length': 0.01,
               'sequence_nce': 1., 'prediction_vjp': 1.}
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in parameters]
    stored = {}
    for name, loss in losses.items():
        gradients = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        if any(g is not None and not torch.isfinite(g).all() for g in gradients):
            raise ValueError('Nonfinite component gradient: ' + name)
        stored[name] = [g.detach().cpu() if g is not None else None for g in gradients]
        del gradients
    total_loss = sum(weights[name] * loss for name, loss in losses.items())
    total = torch.autograd.grad(total_loss, params, allow_unused=True)
    total_cpu = [g.detach().cpu() if g is not None else None for g in total]
    del total
    groups = {'backbone': [], 'content_adapter': [], 'length_head': []}
    for i, (name, _) in enumerate(parameters):
        key = 'backbone' if name.startswith('backbone.') else 'length_head' if name.startswith('adapter.length_head.') else 'content_adapter'
        groups[key].append(i)
    maximum_error, squared_error, total_squared = 0., 0., 0.
    for i, (_, p) in enumerate(parameters):
        summed = sum((stored[key][i].double() * weights[key] for key in losses if stored[key][i] is not None),
                     torch.zeros(p.shape, dtype=torch.float64))
        reference = total_cpu[i].double() if total_cpu[i] is not None else torch.zeros_like(summed)
        difference = summed - reference
        maximum_error = max(maximum_error, float(difference.abs().max()))
        squared_error += float(difference.square().sum())
        total_squared += float(reference.square().sum())
    relative_error = math.sqrt(squared_error / max(total_squared, 1e-30))
    if relative_error > 1e-4:
        raise ValueError('Component sum does not match the independent total derivative')
    report_groups = {}
    for group, indices in groups.items():
        components = {}
        norms = {}
        for key in losses:
            squared = sum(float(stored[key][i].double().square().sum()) for i in indices if stored[key][i] is not None)
            norms[key] = math.sqrt(squared) * weights[key]
            components[key] = dict(unweighted_gradient_norm=math.sqrt(squared), actual_weight=weights[key],
                weighted_gradient_norm=norms[key], tensors_with_gradient=sum(stored[key][i] is not None for i in indices))
        pairs = []
        keys = list(losses)
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                ka, kb = keys[a], keys[b]
                dot = sum(float((stored[ka][i].double() * stored[kb][i].double()).sum())
                          for i in indices if stored[ka][i] is not None and stored[kb][i] is not None)
                dot *= weights[ka] * weights[kb]
                pairs.append(dict(first=ka, second=kb, weighted_dot=dot,
                                  cosine=dot / (norms[ka] * norms[kb]) if norms[ka] * norms[kb] > 0 else None))
        report_groups[group] = dict(parameter_tensors=len(indices), components=components, pairwise=pairs)
    if before != _state_hashes(model) or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Diagnostic changed weights/buffers or accumulated .grad')
    total_norm = math.sqrt(total_squared)
    save_new(args.out, dict(complete=True, checkpoint_sha256=export['checkpoint_sha256'], sample_id=sid,
        source_export_sha256=digest(args.input), source_vjp_sha256=digest(args.vjp),
        original_tail_replay_exact=True, optimizer_updates=0, parameters_and_buffers_unchanged=True,
        prediction_loss_value=vjp['loss'], original_loss_values={key: float(value.detach()) for key, value in losses.items() if key != 'prediction_vjp'},
        prediction_surrogate_value_not_a_loss=True, groups=report_groups,
        component_sum_max_abs_error=maximum_error, component_sum_relative_l2_error=relative_error,
        actual_combined_gradient_norm=total_norm, global_clip_coefficient=min(1., 1. / (total_norm + 1e-6)),
        optimizer_virtual_steps_performed=0, scope='One same-state train example: exact VJP component competition, not Adam contribution or global causal proof'))
    print(json.dumps(dict(event='behavior_gradient_components_finished', total_norm=total_norm,
                         sum_relative_error=relative_error)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['prepare', 'export', 'entry_export', 'entry_compare', 'entry_export_no_tc', 'entry_compare_no_tc', 'h3_vjp', 'music_vjp', 'update', 'gradients'])
    parser.add_argument('--exp_root', type=Path, default=ROOT / 'diff/logs_omnidance/exp')
    parser.add_argument('--weights', type=Path, default=Path('/dev/shm/h3/Ref2VA'))
    parser.add_argument('--cache_list', type=Path, default=Path('/opt/h3cache/posi_list_hr.tsv'))
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--oracle_cache', type=Path, help='Existing same-reference three-joint teacher oracle directory')
    parser.add_argument('--music_python', default='/opt/venv_music/bin/python', help='Existing music proxy interpreter for exact replay')
    parser.add_argument('--legacy_tc_prefix_diagnostic', action='store_true',
        help='Explicitly allow historical entry_compare TC prefix reencoding; not the noTC student protocol')
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--vjp', type=Path)
    parser.add_argument('--optimizer_state', type=Path)
    parser.add_argument('--behavior_weight', type=float, default=0.)
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--latent_frames', type=int, default=2)
    parser.add_argument('--mlp_chunk', type=int, default=0, help='Pointwise token chunking; zero preserves original MLP')
    parser.add_argument('--h3_second_device', default='', help='Move the last half of frozen DiT blocks to a second visible CUDA GPU')
    parser.add_argument('--sigma', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'entry_compare' and not args.legacy_tc_prefix_diagnostic:
        parser.error('Historical TC-prefix entry_compare is disabled by default; it is not the noTC student protocol. Explicit --legacy_tc_prefix_diagnostic required.')
    if args.out.exists() or not args.out.parent.is_dir() or not 0 <= args.index < 4 or not 1 <= args.latent_frames <= 37 or not 0 < args.sigma <= 1 or args.mlp_chunk < 0:
        parser.error('Require new output, valid selected index, latent length and noise level')
    required = dict(prepare=[], export=['checkpoint', 'bundle', 'cache'],
                    entry_export=['checkpoint', 'oracle_cache'], entry_compare=['input'],
                    entry_export_no_tc=['checkpoint', 'oracle_cache'], entry_compare_no_tc=['input'],
                    h3_vjp=['input', 'cache'], music_vjp=['checkpoint', 'input', 'vjp'],
                    update=['checkpoint', 'input', 'cache', 'bundle'],
                    gradients=['checkpoint', 'input', 'bundle', 'vjp'])
    if any(getattr(args, key) is None for key in required[args.stage]):
        parser.error('Missing input path for stage ' + args.stage)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.stage == 'update' and args.behavior_weight and args.vjp is None:
        parser.error('Behavior-supervised update requires a checked H3 VJP')
    {'prepare': prepare, 'export': export_music, 'entry_export': entry_export, 'entry_compare': entry_compare,
     'entry_export_no_tc': entry_export_no_tc, 'entry_compare_no_tc': entry_compare_no_tc, 'h3_vjp': h3_vjp,
     'music_vjp': music_vjp, 'update': update_music, 'gradients': gradient_components}[args.stage](args)


if __name__ == '__main__':
    main()
