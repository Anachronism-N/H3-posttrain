"""Fixed 100-update mean-readout versus permutation-invariant token-readout test."""
import argparse
import hashlib
import json
import random
import tempfile
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from music_text_adapter import (DirectMusicToH3Text, load_direct_music_text_checkpoint,
                                save_direct_music_text_checkpoint)
from music_text_contrastive import (_DirectCPUBackbone, _direct_atomic, _direct_epoch_batches,
    _direct_load_manifest, _direct_losses, _direct_moment_terms, _direct_rng_state,
    _direct_restore_rng, _direct_sample_statistics, _direct_targets, _direct_wave_batch,
    _gradient_scope, score_samples)
from music_text_align import sha256

PROTOCOL = 'direct_mean_vs_token_soft_chamfer_100updates_v1'
INIT_SHA = 'fa7e5ee1ea2d080aa38530ea91ccc29e81f173773618e6c944a3e442618f304e'
MANIFEST_SHA = '40df4ff51bb020a9296ad2c54e9c9c069ef9b1e9710ad2c234d3d7782ea2c6e6'


def raw_teacher(sid, bundle, verified):
    row = bundle['rows'][sid]
    path = row['target_path']
    if path not in verified:
        actual = sha256(path)
        if actual != row['target_sha256']:
            raise ValueError('Used raw teacher SHA mismatch: ' + sid)
        verified[path] = actual
    record = torch.load(path, map_location='cpu', weights_only=True)
    value = record.get('tail')
    if (not isinstance(value, torch.Tensor) or value.shape != (row['teacher_length'], 5120)
            or value.dtype != torch.bfloat16 or not torch.isfinite(value).all()
            or value.requires_grad or record.get('teacher_length') != len(value)):
        raise ValueError('Invalid actual raw teacher tail: ' + sid)
    vector, rms = _direct_sample_statistics(value[None], torch.ones(1, len(value), dtype=torch.bool))
    torch.testing.assert_close(vector[0], bundle['teacher_means'][sid], rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(rms[0], torch.tensor(bundle['teacher_rms'][sid]), rtol=2e-5, atol=2e-5)
    return value


def teacher_batch(ids, bundle, device, verified):
    values = [raw_teacher(sid, bundle, verified) for sid in ids]
    longest = max(map(len, values))
    raw = torch.stack([F.pad(value, (0, 0, 0, longest - len(value))) for value in values]).to(device)
    mask = torch.arange(longest, device=device)[None] < torch.tensor([len(v) for v in values], device=device)[:, None]
    return raw, mask


def token_logits(student, teacher, student_mask, teacher_mask, mean, std, groups):
    if len(student) != len(teacher) or len(groups) != len(student):
        raise ValueError('Token CE requires equal actual paired B rows')
    length = max(student.shape[1], teacher.shape[1])
    x = F.pad(student, (0, 0, 0, length - student.shape[1]))
    y = F.pad(teacher.detach(), (0, 0, 0, length - teacher.shape[1]))
    own = F.pad(student_mask, (0, length - student_mask.shape[1]), value=False)
    valid = F.pad(teacher_mask, (0, length - teacher_mask.shape[1]), value=False)
    # All BxB pairs, each kernel receives distinct student and teacher masks.
    rows = []
    for i in range(len(x)):
        scores = []
        for j in range(len(y)):
            scores.append(score_samples(x[i:i+1], y[j:j+1], valid[j:j+1], mean, std,
                                        limit=64, temperature=0.1, student_mask=own[i:i+1])[0])
        rows.append(torch.stack(scores))
    logits = torch.stack(rows) / 0.07
    group = torch.tensor(groups, device=logits.device)
    excluded = group[:, None].eq(group[None, :])
    excluded.fill_diagonal_(False)
    return logits.masked_fill(excluded, float('-inf'))


def objective(arm, raw, mask, target, target_mask, tm, tr, mean, std, groups, moment_weight):
    sm, sr = _direct_sample_statistics(raw, mask)
    if arm == 'mean':
        loss, terms, _, _ = _direct_losses(sm, sr, tm, tr, mean, std, groups, moment_weight)
        return loss, terms
    logits = token_logits(raw, target, mask, target_mask, mean, std, groups)
    labels = torch.arange(len(logits), device=raw.device)
    clip = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    location, scale = _direct_moment_terms(sm, sr, tm, tr, std)
    moment = (location + scale).mean()
    loss = clip + moment_weight * moment
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite token contrastive/moments loss')
    return loss, dict(clip=clip, moment=moment, location=location.mean(), scale=scale.mean())


def optimizer_for(model):
    return torch.optim.AdamW([
        dict(params=[p for p in model.backbone.parameters() if p.requires_grad], lr=1e-5, name='backbone'),
        dict(params=list(model.adapter.parameters()), lr=1e-4, name='adapter')], weight_decay=0.01)


def fingerprints(args):
    names = [Path(__file__), *[Path(__file__).with_name(name) for name in (
        'music_text_contrastive.py', 'music_text_adapter.py', 'music_text_align.py',
        'music_encoder_adapter.py', 'music_clip_bridge.py')]]
    return dict(sources={str(p.resolve()): sha256(p) for p in names},
                init_sha256=sha256(args.init), manifest_sha256=sha256(args.manifest))


def cpu_test(args):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device('cpu')
    report = dict(protocol=PROTOCOL, cpu_fixture_not_actual_musicfm=True, actual_gpu_training=False)
    x = torch.randn(4, 7, 5120, requires_grad=True)
    y = torch.randn(4, 5, 5120, requires_grad=True)
    own = torch.tensor([[1,0,1,1,0,1,1]] * 4, dtype=torch.bool)
    valid = torch.tensor([[1,1,0,1,1]] * 4, dtype=torch.bool)
    mean, std, groups = torch.randn(5120) * 0.1, torch.rand(5120) + 0.5, [0,0,1,2]
    tm, tr = _direct_sample_statistics(y.detach(), valid)
    sm, sr = _direct_sample_statistics(x, own)
    control, oldterms, _, _ = _direct_losses(sm, sr, tm, tr, mean, std, groups, args.moment_weight)
    actual, terms = objective('mean', x, own, y, valid, tm, tr, mean, std, groups, args.moment_weight)
    assert torch.equal(actual, control) and all(torch.equal(terms[k], oldterms[k]) for k in terms)
    oldgrad = torch.autograd.grad(control, x, retain_graph=True)[0]
    newgrad = torch.autograd.grad(actual, x, retain_graph=True)[0]
    assert torch.equal(oldgrad, newgrad)
    logits = token_logits(x, y, own, valid, mean, std, groups)
    assert logits.shape == (4,4) and torch.isneginf(logits[0,1]) and torch.isneginf(logits[1,0])
    assert torch.isfinite(logits.diagonal()).all() and torch.isfinite(logits[0,2])
    labels = torch.arange(4)
    expected = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    loss, terms = objective('token', x, own, y, valid, tm, tr, mean, std, groups, args.moment_weight)
    assert torch.equal(terms['clip'], expected)
    changed_x, changed_y = x.detach().clone(), y.detach().clone()
    changed_x[~own] = 1000
    changed_y[~valid] = -1000
    torch.testing.assert_close(token_logits(changed_x, changed_y, own, valid, mean, std, groups), logits)
    permutation = torch.tensor([6,3,0,5,2,4,1])
    torch.testing.assert_close(token_logits(x[:,permutation], y, own[:,permutation], valid, mean, std, groups), logits)
    loss.backward()
    assert y.grad is None and torch.count_nonzero(x.grad[~own]) == 0
    report.update(control_loss_and_gradient_exact_old_direct=True, own_mask_padding_invariant=True,
                  teacher_has_no_grad=True, invalid_student_gradient_zero=True,
                  all_BxB_and_bidirectional_CE_and_group_false_negative_mask=True,
                  token_permutation_invariant_not_temporal_supervision=True)
    model = DirectMusicToH3Text(_DirectCPUBackbone(), freeze_backbone=False)
    opt = optimizer_for(model)
    wave = torch.randn(4,2,162000)
    grad_scopes = {}
    for arm in ('mean','token'):
        opt.zero_grad(set_to_none=True)
        raw, mask = model(wave)
        value, _ = objective(arm, raw, mask, y, valid, tm, tr, mean, std, groups, args.moment_weight)
        value.backward()
        grad_scopes[arm] = _gradient_scope(model)
    report['new_entry_fixture_gradient_scopes'] = grad_scopes
    def step(current, optimizer, arm):
        current.train()
        optimizer.zero_grad(set_to_none=True)
        augmented = wave + torch.randn_like(wave) * (random.random() * 0.01)
        raw, mask = current(augmented)
        value, _ = objective(arm, raw, mask, y, valid, tm, tr, mean, std, groups, args.moment_weight)
        value.backward()
        torch.nn.utils.clip_grad_norm_([p for p in current.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step()
        return float(value.detach())
    for arm in ('mean','token'):
        step(model, opt, arm)
        with tempfile.TemporaryDirectory(prefix='new_mapping_resume_') as directory:
            checkpoint = Path(directory) / 'complete.pt'
            state = Path(directory) / 'adam_rng.pt'
            model.eval()
            save_direct_music_text_checkpoint(checkpoint, model)
            torch.save(dict(optimizer=opt.state_dict(), rng=_direct_rng_state(device)), state)
            expected_loss = step(model,opt,arm)
            expected_state = {k:v.clone() for k,v in model.state_dict().items()}
            expected_opt = opt.state_dict()
            fresh = load_direct_music_text_checkpoint(checkpoint, backbone=_DirectCPUBackbone())
            restored = optimizer_for(fresh)
            saved = torch.load(state, map_location='cpu', weights_only=True)
            restored.load_state_dict(saved['optimizer'])
            _direct_restore_rng(saved['rng'], device)
            actual_loss = step(fresh,restored,arm)
            assert expected_loss == actual_loss
            assert all(torch.equal(v,fresh.state_dict()[k]) for k,v in expected_state.items())
            for key, value in expected_opt['state'].items():
                assert all(torch.equal(v,restored.state_dict()['state'][key][k]) for k,v in value.items())
    report['complete_format4_adam_rng_next_step_exact_both_new_objectives'] = True
    hashes = fingerprints(args)
    assert hashes['init_sha256'] == INIT_SHA and hashes['manifest_sha256'] == MANIFEST_SHA
    bundle = _direct_load_manifest(args.manifest, verify_sources=False)
    batches = _direct_epoch_batches(bundle['train'], 8, 1, 42, 1)[:100]
    assert all(len(b)==8 for b in batches) and len({sid for b in batches for sid in b})==800
    report['same_seed42_sampler_100updates_800_unique_exposures'] = dict(
        first_batch=batches[0], sequence_sha256=hashlib.sha256(json.dumps(batches).encode()).hexdigest())
    sid = 'FqlHJVDlFGLKCUJS'
    verified = {}
    actual_tail = raw_teacher(sid, bundle, verified)
    assert actual_tail.shape == (234,5120)
    report['actual_lazy_one_raw_teacher'] = dict(sample_id=sid, shape=list(actual_tail.shape),
        dtype=str(actual_tail.dtype), finite=True, SHA256=verified[bundle['rows'][sid]['target_path']],
        full_6500_raw_teacher_loaded=False, manifest_moments_equivalent=True)
    report.update(complete=True, moment_weight=args.moment_weight, source_fingerprints=hashes)
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report), flush=True)


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError('GPU training must use actual MusicFM')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    random.seed(42)
    torch.cuda.set_device(0)
    device = torch.device('cuda',0)
    fp = fingerprints(args)
    if fp['init_sha256'] != INIT_SHA or fp['manifest_sha256'] != MANIFEST_SHA:
        raise ValueError('Locked init/manifest SHA differs')
    if not args.cpu_report:
        raise ValueError('New entry CPU feature report required')
    tested = json.loads(args.cpu_report.read_text())
    if (not tested.get('complete') or tested.get('protocol') != PROTOCOL
            or tested.get('source_fingerprints') != fp or tested.get('moment_weight') != args.moment_weight):
        raise ValueError('Actual new feature CPU report mismatch')
    bundle = _direct_load_manifest(args.manifest, verify_sources=False)
    batches = _direct_epoch_batches(bundle['train'],8,1,42,1)[:100]
    config = dict(arm=args.arm, steps=100, local_batch=8, global_batch=8, seed=42,
        backbone_lr=1e-5, adapter_lr=1e-4, weight_decay=0.01, gradient_clip=1.,
        moment_weight=args.moment_weight, optimizer='fresh_AdamW_no_schedule',
        token_kernel_limit=64, token_kernel_temperature=0.1, CE_temperature=0.07)
    owned, history, verified, steps = set(), [], {}, 0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=True)
        if (saved['protocol'] != PROTOCOL or saved['configuration'] != config
                or saved['fingerprints'] != fp or saved['batches'] != batches
                or saved['out'] != str(args.out.resolve())
                or sha256(args.out/'last.pt') != saved['checkpoint_sha256']):
            raise ValueError('Complete model/Adam/RNG/sampler resume mismatch')
        model = load_direct_music_text_checkpoint(args.out/'last.pt').to(device)
        steps, history, owned, verified = saved['steps'], saved['history'], set(saved['owned']), saved['verified_sources']
    else:
        if args.out.exists():
            raise FileExistsError('Refuse existing fresh experiment output')
        args.out.mkdir()
        model = load_direct_music_text_checkpoint(args.init).to(device)
    model.configure_online_training(True)
    if model.freeze_backbone:
        raise ValueError('Both arms must train complete dense MusicFM scope')
    optimizer = optimizer_for(model)
    if args.resume:
        optimizer.load_state_dict(saved['optimizer'])
        _direct_restore_rng(saved['rng'], device)
    buffers = {k:v.detach().cpu().clone() for k,v in model.named_buffers()}
    plan = dict(protocol=PROTOCOL, configuration=config, fingerprints=fp,
        first_batch=batches[0], sampler_sha256=hashlib.sha256(json.dumps(batches).encode()).hexdigest(),
        required_actual_updates=100, required_exposures=800, best_selection=False, early_stop=False,
        cached_features=False, student_inputs='only_stereo_32k_162000_wave', cached_prefix_loaded=False,
        raw_condition_deploy='uncropped_unnormalized_channel_major_[254,5120]',
        raw_teacher_loading='actual_per_sample_lazy_BF16_no_full_6500_no_reencoding',
        source_verification='locked_init_manifest_and_each_used_audio_target_prefix_once',
        deployment='cached_P_tags_VAE_and_native_true_music_original_AV_joint',
        scientific_scope='mean_readout_loses_token_distribution; softChamfer_permutation_invariant_not_time_supervision')
    if not args.resume:
        _direct_atomic(args.out/'plan.json',plan,owned)
    started = time.monotonic()
    def checkpoint():
        model.eval()
        _direct_atomic(args.out/'last.pt',None,owned,checkpoint_model=model)
        owned.update({'train_state.pt'})
        state = dict(protocol=PROTOCOL, configuration=config, fingerprints=fp, batches=batches,
            out=str(args.out.resolve()), steps=steps, history=history, owned=sorted(owned),
            optimizer=optimizer.state_dict(), rng=_direct_rng_state(device), verified_sources=verified,
            checkpoint_sha256=sha256(args.out/'last.pt'))
        _direct_atomic(args.out/'train_state.pt',state,owned)
    try:
        for ids in batches[steps:]:
            print(json.dumps(dict(event='loading_actual_batch', next_update=steps+1, ids=ids)),flush=True)
            for sid in ids:
                row = bundle['rows'][sid]
                for kind in ('audio','prefix'):
                    path = row[kind+'_path']
                    if path not in verified:
                        actual = sha256(path)
                        if actual != row[kind+'_sha256']:
                            raise ValueError('Used sample source SHA mismatch: '+sid+':'+kind)
                        verified[path] = actual
            target, target_mask = teacher_batch(ids,bundle,device,verified)
            tm,tr,groups = _direct_targets(ids,bundle,device)
            wave = _direct_wave_batch(ids,bundle,device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            raw, mask = model(wave)
            if raw.shape != (8,254,5120) or not mask.all():
                raise ValueError('Actual fullwave MusicFM must be 127 frames per stereo channel')
            loss, terms = objective(args.arm, raw, mask, target, target_mask, tm,tr,
                                   bundle['mean'].to(device),bundle['std'].to(device),groups,args.moment_weight)
            loss.backward()
            if steps == 0:
                gradient = _gradient_scope(model)
                if (gradient['backbone']['parameter_tensors'] != 400
                        or gradient['backbone']['parameter_elements'] != 316343296
                        or gradient['adapter']['parameter_tensors'] != 2
                        or gradient['adapter']['parameter_elements'] != 5248000):
                    raise ValueError('Actual complete dense/linear gradient scope differs')
                gradient.update(actual_full_MusicFM=True, raw_shape=list(raw.shape),
                                cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                                cuda_peak_reserved=torch.cuda.max_memory_reserved())
                _direct_atomic(args.out/'gradient_step1.json',gradient,owned)
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            record = dict(update=steps, ids=ids, actual_exposures=steps*8,
                loss=float(loss.detach()), gradient_norm_preclip=float(norm),
                **{k:float(v.detach()) for k,v in terms.items()},
                cuda_peak_allocated=torch.cuda.max_memory_allocated(), seconds=time.monotonic()-started)
            history.append(record)
            print(json.dumps(dict(event='actual_successful_update', **record)),flush=True)
            del raw,mask,loss,terms,wave,target,target_mask
            if steps == 1 or steps % 25 == 0:
                checkpoint()
        for key,value in model.named_buffers():
            if not torch.equal(value.detach().cpu(),buffers[key]):
                raise ValueError('Fixed backbone buffer changed: '+key)
        if steps != 100 or sum(len(r['ids']) for r in history)!=800:
            raise ValueError('Actual successful update/exposure protocol mismatch')
        final = torch.load(args.out/'last.pt',map_location='cpu',weights_only=True)
        if len(final['model']) != 467 or final['metadata']['format_version'] != 4:
            raise ValueError('Final deployed checkpoint is not complete 467-state format4')
        summary = dict(protocol=PROTOCOL, complete=True, actual_successful_updates=steps,
            actual_exposures=800, checkpoint_sha256=sha256(args.out/'last.pt'),
            train_state_sha256=sha256(args.out/'train_state.pt'), final_format4_state_tensors=467,
            configuration=config, raw_teacher_lazy=True, verified_used_sources=verified,
            native_joint_three_case_inference_pending_N4=True, video_quality_not_accepted=True)
        _direct_atomic(args.out/'finished.json',summary,owned)
        print(json.dumps(dict(event='fixed_mapping_finished',**summary)),flush=True)
    except BaseException as error:
        # Last successful checkpoint is retained; never relabel partial as completed.
        failure = dict(protocol=PROTOCOL,complete=False,actual_successful_updates=steps,
                       actual_exposures=steps*8,error=repr(error),history=history)
        _direct_atomic(args.out/'partial_failure.json',failure,owned)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['mean','token'],default='mean')
    parser.add_argument('--init',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--moment_weight',type=float,required=True)
    parser.add_argument('--cpu_test',action='store_true')
    parser.add_argument('--cpu_report',type=Path)
    parser.add_argument('--resume',type=Path)
    args = parser.parse_args()
    (cpu_test if args.cpu_test else train)(args)


if __name__ == '__main__':
    main()
