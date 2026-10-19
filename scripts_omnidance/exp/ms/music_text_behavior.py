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
    parser.add_argument('--stage', required=True, choices=['prepare', 'export', 'h3_vjp', 'music_vjp', 'update', 'gradients'])
    parser.add_argument('--exp_root', type=Path, default=ROOT / 'diff/logs_omnidance/exp')
    parser.add_argument('--weights', type=Path, default=Path('/dev/shm/h3/Ref2VA'))
    parser.add_argument('--cache_list', type=Path, default=Path('/opt/h3cache/posi_list_hr.tsv'))
    parser.add_argument('--cache', type=Path)
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
    if args.out.exists() or not args.out.parent.is_dir() or not 0 <= args.index < 4 or not 1 <= args.latent_frames <= 37 or not 0 < args.sigma <= 1 or args.mlp_chunk < 0:
        parser.error('Require new output, valid selected index, latent length and noise level')
    required = dict(prepare=[], export=['checkpoint', 'bundle', 'cache'],
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
    {'prepare': prepare, 'export': export_music, 'h3_vjp': h3_vjp,
     'music_vjp': music_vjp, 'update': update_music, 'gradients': gradient_components}[args.stage](args)


if __name__ == '__main__':
    main()
