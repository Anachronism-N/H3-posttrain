"""Complete 6000/500, three-epoch continuation of the unchanged token objective."""
import argparse
import hashlib
import json
import math
import random
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F
import music_direct_token_compare as original
from music_direct_token_compare import (INIT_SHA, MANIFEST_SHA, objective, optimizer_for,
                                        raw_teacher, teacher_batch)
from music_text_adapter import (DirectMusicToH3Text, load_direct_music_text_checkpoint,
                                save_direct_music_text_checkpoint)
from music_text_contrastive import (_DirectCPUBackbone, _direct_atomic, _direct_epoch_batches,
    _direct_moment_terms, _direct_rng_state, _direct_restore_rng,
    _direct_sample_statistics, _direct_targets, _direct_wave_batch, _gradient_scope)
from music_text_align import sha256
from music_text_contrastive import DIRECT_CLIP_PROTOCOL

PROTOCOL = 'direct_token_full6000_val500_epochs3_v1'
STEPS = 2250
EPOCH_STEPS = 750
JOINT = Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/music_direct_eval3_generation_joint_v1.json')
IDS = ['wJPptuhclZmKAEWs', 'wJPrpFsThWAcBQIw', 'wJQTclrlOLEYYfTB']


def load_manifest(path, verify_sources=False):
    manifest = path if isinstance(path, dict) else json.loads(Path(path).read_text())
    if not isinstance(manifest, dict) or set(manifest) != {'metadata', 'rows', 'statistics'}:
        raise ValueError('Direct manifest requires metadata/rows/statistics only')
    metadata, rows, statistics = (manifest[key] for key in ('metadata', 'rows', 'statistics'))
    if (not isinstance(metadata, dict) or metadata.get('protocol') != DIRECT_CLIP_PROTOCOL
            or metadata.get('complete') is not True or not isinstance(rows, dict)
            or not isinstance(statistics, dict)):
        raise ValueError('Direct manifest is not a completed direct-token release')
    train, val = metadata.get('train_ids'), metadata.get('val_ids')
    if (not isinstance(train, list) or not isinstance(val, list) or len(train) != 6000 or len(val) != 500
            or any(type(sid) is not str or not sid for sid in train + val)
            or len(set(train + val)) != 6500 or set(rows) != set(train + val)):
        raise ValueError('Require the locked 6000/500 disjoint unique sample split')
    mean, std = (torch.tensor(statistics.get(key, []), dtype=torch.float32) for key in ('mean', 'std'))
    energy, token_count = statistics.get('teacher_energy'), statistics.get('training_token_count')
    if (mean.shape != (5120,) or std.shape != (5120,) or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all() or (std <= 0).any()
            or type(energy) not in (float, int) or not math.isfinite(energy) or energy <= 0
            or type(token_count) is not int or token_count < 6000):
        raise ValueError('Invalid fixed full-train raw-token statistics')
    teacher_means, teacher_rms, group_indices = {}, {}, {}
    groups = {}
    train_groups, val_groups = set(), set()
    required = {'audio_path', 'audio_sha256', 'group_id', 'target_path', 'target_sha256',
                'prefix_path', 'prefix_sha256', 'partition', 'teacher_mean',
                'teacher_centered_rms', 'teacher_length'}
    train_tokens = 0
    verified = {}
    for partition, ids, partition_groups in (('train', train, train_groups), ('val', val, val_groups)):
        for sid in ids:
            row = rows[sid]
            if (not isinstance(row, dict) or set(row) != required or row['partition'] != partition
                    or type(row['group_id']) is not str or not row['group_id']
                    or type(row['teacher_length']) is not int or row['teacher_length'] < 1
                    or type(row['teacher_centered_rms']) not in (int, float)
                    or not math.isfinite(row['teacher_centered_rms']) or row['teacher_centered_rms'] < 0):
                raise ValueError('Invalid direct teacher table row: ' + sid)
            vector = torch.tensor(row['teacher_mean'], dtype=torch.float32)
            if vector.shape != (5120,) or not torch.isfinite(vector).all():
                raise ValueError('Require the full finite raw-y teacher mean: ' + sid)
            teacher_means[sid], teacher_rms[sid] = vector, float(row['teacher_centered_rms'])
            group = row['group_id']
            groups.setdefault(group, len(groups))
            group_indices[sid] = groups[group]
            partition_groups.add(group)
            if partition == 'train':
                train_tokens += row['teacher_length']
            for kind in ('audio', 'target', 'prefix'):
                source, checksum = Path(row[kind + '_path']), row[kind + '_sha256']
                if (type(checksum) is not str or len(checksum) != 64
                        or any(c not in '0123456789abcdef' for c in checksum)):
                    raise ValueError('Invalid manifest source SHA: ' + sid + ':' + kind)
                if kind == 'audio' and source.suffix != '.wav':
                    raise ValueError('Deployment load_audio requires its actual .wav source')
                if not source.is_absolute() or '..' in source.parts:
                    raise ValueError('Locked sources must be absolute canonical paths')
                key = str(source)
                if key in verified and verified[key] != checksum:
                    raise ValueError('Conflicting source SHA in manifest: ' + key)
                verified[key] = checksum
    if train_groups & val_groups or train_tokens != token_count:
        raise ValueError('Train/validation song groups overlap or training token count differs')
    if verify_sources:
        raise ValueError('Full6k verifies sources only when consumed')
    return dict(metadata=metadata, rows=rows, train=train, val=val, mean=mean, std=std,
                teacher_means=teacher_means, teacher_rms=teacher_rms, groups=group_indices,
                teacher_energy=float(energy), source_files=verified)




def sampler(bundle):
    epochs = [_direct_epoch_batches(bundle['train'], 8, 1, 42, epoch) for epoch in range(1, 4)]
    for batches in epochs:
        assert len(batches) == EPOCH_STEPS and all(len(batch) == 8 for batch in batches)
        assert len({sid for batch in batches for sid in batch}) == 6000
        assert {sid for batch in batches for sid in batch} == set(bundle['train'])
        assert not {sid for batch in batches for sid in batch} & set(bundle['val'])
    assert epochs[0] != epochs[1] != epochs[2]
    return [batch for batches in epochs for batch in batches]


def fingerprints(args):
    hashes = original.fingerprints(args)
    hashes['sources'][str(Path(__file__).resolve())] = sha256(__file__)
    hashes['eval3_joint_sha256'] = sha256(JOINT)
    return hashes


def cpu_test(args):
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='full6k_acceptance_') as temporary:
        old_report = Path(temporary) / 'original.json'
        original_loader = original._direct_load_manifest
        try:
            original._direct_load_manifest = load_manifest
            original.cpu_test(SimpleNamespace(init=args.init, manifest=args.manifest,
                moment_weight=args.moment_weight, out=old_report))
        finally:
            original._direct_load_manifest = original_loader
        inherited = json.loads(old_report.read_text())
    bundle = load_manifest(args.manifest, verify_sources=False)
    batches = sampler(bundle)
    assert len(batches) == STEPS and sum(map(len, batches)) == 18000
    joint = json.loads(JOINT.read_text())
    assert joint['condition'] == 'joint' and [row['sample_id'] for row in joint['cases']] == IDS
    assert not set(IDS) & set(bundle['train'] + bundle['val'])
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device('cpu')
    model = DirectMusicToH3Text(_DirectCPUBackbone(), freeze_backbone=False)
    optimizer = optimizer_for(model)
    teacher = torch.randn(3, 5, 5120)
    teacher_mask = torch.ones(3, 5, dtype=torch.bool)
    tm, tr = _direct_sample_statistics(teacher, teacher_mask)
    mean, std = torch.zeros(5120), torch.ones(5120)
    def step(current, opt, cursor):
        ids = batches[cursor]
        opt.zero_grad(set_to_none=True)
        wave = torch.randn(3, 2, 162000) * (1 + random.random())
        raw, mask = current(wave)
        loss, _ = objective('token', raw, mask, teacher, teacher_mask, tm, tr,
                            mean, std, [0, 1, 2], args.moment_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in current.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        opt.step()
        return float(loss.detach()), ids
    step(model, optimizer, 748)
    with tempfile.TemporaryDirectory(prefix='full6k_resume_') as directory:
        checkpoint = Path(directory) / 'complete.pt'
        state_path = Path(directory) / 'state.pt'
        model.eval()
        save_direct_music_text_checkpoint(checkpoint, model)
        torch.save(dict(optimizer=optimizer.state_dict(), rng=_direct_rng_state(device),
            batches=batches, steps=749, budget=STEPS), state_path)
        expected_loss, expected_ids = step(model, optimizer, 749)
        expected_state = {key:value.clone() for key,value in model.state_dict().items()}
        expected_optimizer = optimizer.state_dict()
        expected_next_epoch = batches[750]
        restored = load_direct_music_text_checkpoint(checkpoint, backbone=_DirectCPUBackbone())
        restored_optimizer = optimizer_for(restored)
        saved = torch.load(state_path, map_location='cpu', weights_only=True)
        assert saved['budget'] == STEPS and saved['batches'] == batches
        restored_optimizer.load_state_dict(saved['optimizer'])
        _direct_restore_rng(saved['rng'], device)
        actual_loss, actual_ids = step(restored, restored_optimizer, saved['steps'])
        assert actual_loss == expected_loss and actual_ids == expected_ids
        assert saved['batches'][saved['steps'] + 1] == expected_next_epoch
        assert all(torch.equal(value, restored.state_dict()[key]) for key,value in expected_state.items())
        for key, values in expected_optimizer['state'].items():
            assert all(torch.equal(value, restored_optimizer.state_dict()['state'][key][name])
                       for name,value in values.items())
    # Vectorized all-pairs validation must equal the original masked 64-token kernel.
    x = torch.randn(3, 7, 5120)
    y = torch.randn(3, 5, 5120)
    xm = torch.ones(3, 7, dtype=torch.bool)
    ym = torch.tensor([[1,1,0,1,1], [1,0,1,0,0], [1,1,1,1,1]], dtype=torch.bool)
    xs, xvalid = sampled_tokens(x, xm, mean, std)
    ys, yvalid = sampled_tokens(y, ym, mean, std)
    expected = original.token_logits(x, y, xm, ym, mean, std, [0,1,2]) * .07
    actual = set_score_matrix(xs, ys, xvalid, yvalid)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    hashes = fingerprints(args)
    assert hashes['init_sha256'] == INIT_SHA and hashes['manifest_sha256'] == MANIFEST_SHA
    report = dict(protocol=PROTOCOL, complete=True, actual_gpu_training=False,
        inherited_actual_new_objective_checks=inherited,
        three_epochs_each_6000_unique_no_repeat=True, epoch_seeds=[43,44,45],
        train_val_disjoint=True, updates=STEPS, exposures=18000, global_batch=8,
        sampler_sha256=hashlib.sha256(json.dumps(batches).encode()).hexdigest(),
        full_format4_Adam_RNG_sampler_next_step_exact=True,
        epoch_boundary_next_batch_and_remaining_budget_exact=True,
        vectorized_val_token_kernel_matches_original=True, fixed3_extra_holdouts=IDS,
        moment_weight=args.moment_weight, source_fingerprints=hashes)
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(event='full6k_cpu_acceptance_complete', report=str(args.out), updates=STEPS)), flush=True)


def verify_used(ids, bundle, verified):
    for sid in ids:
        row = bundle['rows'][sid]
        for kind in ('audio', 'prefix'):
            path = row[kind + '_path']
            if path not in verified:
                actual = sha256(path)
                if actual != row[kind + '_sha256']:
                    raise ValueError('Used source SHA mismatch: ' + sid + ':' + kind)
                verified[path] = actual


def sampled_tokens(raw, mask, mean, std):
    values, masks = [], []
    for value, valid in zip(raw, mask):
        indices = valid.nonzero().flatten()
        if len(indices) > 64:
            indices = indices[torch.linspace(0, len(indices)-1, 64, device=indices.device).round().long()]
        selected = F.normalize((value.index_select(0, indices).float()-mean)/std, dim=-1)
        values.append(F.pad(selected, (0,0,0,64-len(selected))))
        masks.append(torch.arange(64, device=value.device) < len(selected))
    return torch.stack(values), torch.stack(masks)


def set_score_matrix(x, y, xm, ym):
    result = x.new_empty((len(x), len(y)))
    for i in range(0, len(x), 8):
        for j in range(0, len(y), 16):
            a, b = x[i:i+8], y[j:j+16]
            own, valid = xm[i:i+8], ym[j:j+16]
            sim = (a.flatten(0,1) @ b.flatten(0,1).T).reshape(len(a),64,len(b),64).permute(0,2,1,3)
            row = .1 * (torch.logsumexp((sim/.1).masked_fill(~valid[None,:,None,:], float('-inf')),dim=-1)
                        - valid.sum(-1).log()[None,:,None])
            col = .1 * (torch.logsumexp((sim/.1).masked_fill(~own[:,None,:,None], float('-inf')),dim=-2)
                        - own.sum(-1).log()[:,None,None])
            result[i:i+len(a),j:j+len(b)] = .5 * (
                (row * own[:,None,:]).sum(-1) / own.sum(-1)[:,None]
                + (col * valid[None,:,:]).sum(-1) / valid.sum(-1)[None,:])
    return result


def retrieval_metrics(scores):
    labels = torch.arange(len(scores), device=scores.device)
    result = dict(candidates=len(scores), bidirectional_CE=float(.5 * (
        F.cross_entropy(scores/.07, labels) + F.cross_entropy(scores.T/.07, labels))))
    for direction, matrix in [('music_to_teacher',scores), ('teacher_to_music',scores.T)]:
        result[direction] = {'R@' + str(k):float((matrix.topk(k, dim=1).indices == labels[:,None]).any(1).float().mean())
                             for k in (1,5,10)}
    return result


@torch.no_grad()
def evaluate(model, bundle, device, verified, args, epoch, owned):
    model.eval()
    ids = bundle['val']
    mean, std = bundle['mean'].to(device), bundle['std'].to(device)
    students, teachers, smasks, tmasks, means, rms_values = [], [], [], [], [], []
    started = time.monotonic()
    for start in range(0, 500, 8):
        names = ids[start:start+8]
        verify_used(names, bundle, verified)
        wave = _direct_wave_batch(names, bundle, device)
        raw, mask = model(wave)
        if raw.shape != (len(names),254,5120) or not mask.all():
            raise ValueError('Validation raw shape differs')
        target, valid = teacher_batch(names, bundle, device, verified)
        xs, xm = sampled_tokens(raw, mask, mean, std)
        ys, ym = sampled_tokens(target, valid, mean, std)
        sm, sr = _direct_sample_statistics(raw, mask)
        students.append(xs); teachers.append(ys); smasks.append(xm); tmasks.append(ym)
        means.append(sm); rms_values.append(sr)
        Path('/tmp/od_reserve_0').touch()
    sm, sr = torch.cat(means), torch.cat(rms_values)
    tm, tr, groups = _direct_targets(ids, bundle, device)
    token_scores = set_score_matrix(torch.cat(students), torch.cat(teachers), torch.cat(smasks), torch.cat(tmasks))
    whitened_student, whitened_teacher = F.normalize((sm-mean)/std, dim=-1), F.normalize((tm-mean)/std, dim=-1)
    mean_scores = whitened_student @ whitened_teacher.T
    group = torch.tensor(groups, device=device)
    excluded = group[:,None].eq(group[None,:]); excluded.fill_diagonal_(False)
    location, scale = _direct_moment_terms(sm, sr, tm, tr, std)
    metrics = dict(epoch=epoch, validation_samples=500, optimizer_updates=epoch*EPOCH_STEPS,
        token_set_retrieval_500=retrieval_metrics(token_scores.masked_fill(excluded,float('-inf'))),
        mean_readout_retrieval_500=retrieval_metrics(mean_scores.masked_fill(excluded,float('-inf'))),
        paired_metrics=dict(mean_cosine=float(mean_scores.diagonal().mean()),
            token_softchamfer64=float(token_scores.diagonal().mean()),
            location=float(location.mean()), scale=float(scale.mean()),
            moment=float((location+scale).mean()), student_centered_rms=float(sr.mean()),
            teacher_centered_rms=float(tr.mean())),
        group_false_negative_pairs_masked=int(excluded.sum()), seconds=time.monotonic()-started,
        selection='none_fixed3epochs_no_early_stop_no_schedule',
        scientific_scope='token_set_is_permutation_invariant_not_temporal_or_video_quality')
    _direct_atomic(args.out / ('validation_epoch%02d.json' % epoch), metrics, owned)
    joint = json.loads(JOINT.read_text())
    assert joint['condition']=='joint' and [row['sample_id'] for row in joint['cases']]==IDS
    from music_clip_bridge import load_audio
    for row in joint['cases']:
        path = row['audio_path']
        if path not in verified:
            actual = sha256(path)
            if actual != row['audio_sha256']:
                raise ValueError('Fixed3 source SHA differs')
            verified[path] = actual
        source = Path(path)
        wave = load_audio(source.stem, source.parent)[0]
        assert wave.shape==(2,162000)
        raw, mask = model(wave[None].to(device))
        assert raw.shape==(1,254,5120) and mask.all() and raw.dtype==torch.float32
        _direct_atomic(args.out / ('epoch%02d_%s_rawhidden.pt' % (epoch,row['sample_id'])),
            dict(raw=raw[0].cpu(), mask=mask[0].cpu(), sample_id=row['sample_id'],
                epoch=epoch, actual_successful_updates=epoch*EPOCH_STEPS,
                audio_sha256=row['audio_sha256'], student_inputs='wave_only',
                output='uncropped_unnormalized_channel_major_254x5120',
                teacher_or_prefix_loaded_into_student=False, video_generated=False,
                autoregressive_decoder_invoked=False), owned)
    print(json.dumps(dict(event='epoch_validation_and_fixed3_hidden_complete', **metrics)),flush=True)
    return metrics


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError('Actual full6k training requires MusicFM CUDA')
    torch.set_num_threads(2)
    torch.manual_seed(42); random.seed(42)
    torch.cuda.set_device(0)
    device = torch.device('cuda',0)
    fp = fingerprints(args)
    if fp['init_sha256'] != INIT_SHA or fp['manifest_sha256'] != MANIFEST_SHA:
        raise ValueError('Locked initial checkpoint/manifest differs')
    tested = json.loads(args.cpu_report.read_text())
    if (tested.get('complete') is not True or tested.get('protocol') != PROTOCOL
            or tested.get('source_fingerprints') != fp or tested.get('moment_weight') != args.moment_weight):
        raise ValueError('Actual new CPU feature report mismatch')
    bundle = load_manifest(args.manifest, verify_sources=False)
    batches = sampler(bundle)
    config = dict(arm='token',steps=STEPS,epochs=3,train_samples=6000,val_samples=500,
        local_batch=8,global_batch=8,seed=42,epoch_seeds=[43,44,45],backbone_lr=1e-5,
        adapter_lr=1e-4,weight_decay=.01,gradient_clip=1.,moment_weight=args.moment_weight,
        optimizer='fresh_AdamW_no_schedule',token_kernel_limit=64,token_kernel_temperature=.1,CE_temperature=.07)
    owned, history, verified, steps, evaluations = set(), [], {}, 0, []
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=True)
        if (saved['protocol']!=PROTOCOL or saved['configuration']!=config or saved['fingerprints']!=fp
                or saved['batches']!=batches or saved['out']!=str(args.out.resolve())
                or sha256(args.out/saved['checkpoint_name'])!=saved['checkpoint_sha256']):
            raise ValueError('Complete full6k model/Adam/RNG/sampler/budget resume mismatch')
        model = load_direct_music_text_checkpoint(args.out/saved['checkpoint_name']).to(device)
        steps, history, owned, verified, evaluations = (saved['steps'],saved['history'],set(saved['owned']),
            saved['verified_sources'],saved['evaluations'])
    else:
        if args.out.exists():
            raise FileExistsError('New dedicated output must not exist')
        args.out.mkdir()
        model = load_direct_music_text_checkpoint(args.init).to(device)
    model.configure_online_training(True)
    if model.freeze_backbone:
        raise ValueError('Complete dense backbone must remain trainable')
    optimizer = optimizer_for(model)
    if args.resume:
        optimizer.load_state_dict(saved['optimizer']); _direct_restore_rng(saved['rng'],device)
    buffers = {key:value.detach().cpu().clone() for key,value in model.named_buffers()}
    if not args.resume:
        _direct_atomic(args.out/'plan.json',dict(protocol=PROTOCOL,configuration=config,fingerprints=fp,
            required_actual_updates=STEPS,required_exposures=18000,required_unique_train=6000,
            first_batch=batches[0],sampler_sha256=hashlib.sha256(json.dumps(batches).encode()).hexdigest(),
            original_loss_source=str(Path(original.__file__).resolve()),original_100update_outputs_untouched=True,
            raw_condition_deploy='uncropped_unnormalized_channel_major_[254,5120]',student_inputs='wave_only',
            cached_features=False,cached_prefix_loaded=False,teacher='supervision_only_lazy_BF16_actual_raw',
            verification='each_actually_consumed_audio_target_prefix_once_cached_through_resume',
            checkpoint_interval='500updates_and_each_epoch_immutable_full_model_and_Adam_RNG_sampler',
            evaluation='500val_bidirectional_token_and_mean_retrieval_distinct_paired_metrics_plus3_hidden_no_video',
            best_selection=False,early_stop=False),owned)
    started = time.monotonic()
    def checkpoint():
        model.eval()
        name = 'model_step%05d.pt' % steps
        state_name = 'train_state_step%05d.pt' % steps
        _direct_atomic(args.out/name,None,owned,checkpoint_model=model)
        owned.update({state_name, 'latest.json'})
        state = dict(protocol=PROTOCOL,configuration=config,fingerprints=fp,batches=batches,
            out=str(args.out.resolve()),steps=steps,remaining_updates=STEPS-steps,
            next_epoch=steps//EPOCH_STEPS+1,next_batch_in_epoch=steps%EPOCH_STEPS,
            history=history,owned=sorted(owned),optimizer=optimizer.state_dict(),rng=_direct_rng_state(device),
            verified_sources=verified,evaluations=evaluations,checkpoint_name=name,
            checkpoint_sha256=sha256(args.out/name))
        _direct_atomic(args.out/state_name,state,owned)
        _direct_atomic(args.out/'latest.json',dict(steps=steps,checkpoint_name=name,train_state_name=state_name,
            checkpoint_sha256=state['checkpoint_sha256']),owned)
        print(json.dumps(dict(event='complete_immutable_checkpoint',steps=steps,model=name,state=state_name)),flush=True)
    try:
        # A pre-epoch-validation checkpoint can be resumed without skipping that evaluation.
        if steps and steps%EPOCH_STEPS==0 and len(evaluations)<steps//EPOCH_STEPS:
            evaluations.append(evaluate(model,bundle,device,verified,args,steps//EPOCH_STEPS,owned))
        for ids in batches[steps:]:
            print(json.dumps(dict(event='loading_actual_batch',next_update=steps+1,
                epoch=steps//EPOCH_STEPS+1,ids=ids)),flush=True)
            verify_used(ids,bundle,verified)
            target,target_mask = teacher_batch(ids,bundle,device,verified)
            tm,tr,groups = _direct_targets(ids,bundle,device)
            wave = _direct_wave_batch(ids,bundle,device)
            model.train(); optimizer.zero_grad(set_to_none=True)
            raw,mask = model(wave)
            if raw.shape!=(8,254,5120) or not mask.all():
                raise ValueError('Actual complete raw waveform output differs')
            loss,terms = objective('token',raw,mask,target,target_mask,tm,tr,
                bundle['mean'].to(device),bundle['std'].to(device),groups,args.moment_weight)
            loss.backward()
            if steps==0:
                gradient = _gradient_scope(model)
                if (gradient['backbone']['parameter_tensors']!=400 or gradient['backbone']['parameter_elements']!=316343296
                        or gradient['adapter']['parameter_tensors']!=2 or gradient['adapter']['parameter_elements']!=5248000):
                    raise ValueError('First budgeted update full dense gradient differs')
                gradient.update(actual_full_MusicFM=True,raw_shape=list(raw.shape),first_update_in_2250_budget=True)
                _direct_atomic(args.out/'gradient_step1.json',gradient,owned)
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step(); steps += 1
            record = dict(update=steps,epoch=(steps-1)//EPOCH_STEPS+1,ids=ids,actual_exposures=steps*8,
                loss=float(loss.detach()),gradient_norm_preclip=float(norm),
                **{key:float(value.detach()) for key,value in terms.items()},
                cuda_peak_allocated=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started)
            history.append(record)
            _direct_atomic(args.out/'progress.json',dict(protocol=PROTOCOL,complete=False,**record,
                actual_unique_train=len({sid for event in history for sid in event['ids']})),owned)
            print(json.dumps(dict(event='actual_successful_update',**record)),flush=True)
            del raw,mask,loss,terms,wave,target,target_mask
            Path('/tmp/od_reserve_0').touch()
            if steps%EPOCH_STEPS==0:
                evaluations.append(evaluate(model,bundle,device,verified,args,steps//EPOCH_STEPS,owned))
            if steps%500==0 or steps%EPOCH_STEPS==0:
                checkpoint()
        for key,value in model.named_buffers():
            if not torch.equal(value.detach().cpu(),buffers[key]):
                raise ValueError('Frozen persistent buffer changed: '+key)
        if steps!=STEPS or sum(len(event['ids']) for event in history)!=18000 or len(evaluations)!=3:
            raise ValueError('Actual update/exposure/evaluation budget incomplete')
        final = torch.load(args.out/('model_step%05d.pt'%STEPS),map_location='cpu',weights_only=True)
        if len(final['model'])!=467 or final['metadata']['format_version']!=4:
            raise ValueError('Full deployed format4 state incomplete')
        _direct_atomic(args.out/'finished.json',dict(protocol=PROTOCOL,complete=True,
            actual_successful_updates=steps,actual_exposures=18000,actual_unique_train=6000,
            validation_epochs=3,configuration=config,final_format4_state_tensors=467,
            final_checkpoint=str(args.out/('model_step%05d.pt'%STEPS)),
            video_quality_accepted=False,native_AV_joint_video_only_N4=True),owned)
    except BaseException as error:
        _direct_atomic(args.out/'partial_failure.json',dict(protocol=PROTOCOL,complete=False,
            actual_successful_updates=steps,actual_exposures=steps*8,error=repr(error),history=history),owned)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--moment_weight',type=float,default=1.)
    parser.add_argument('--cpu_test',action='store_true')
    parser.add_argument('--cpu_report',type=Path)
    parser.add_argument('--resume',type=Path)
    args = parser.parse_args()
    (cpu_test if args.cpu_test else train)(args)


if __name__=='__main__':
    main()
