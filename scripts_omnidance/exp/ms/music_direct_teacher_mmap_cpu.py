"""Read-only actual-target mmap equality test; never launches training."""
import argparse
import copy
import hashlib
import json
import mmap
import os
import random
import struct
import sys
import zipfile
from pathlib import Path

import torch
from torch.nn import functional as F
from torch._subclasses.fake_tensor import FakeTensorMode
import music_direct_token_compare as original
import music_direct_token_full6k as full
from music_text_align import sha256
from music_text_contrastive import _direct_sample_statistics, _direct_targets


def raw_teacher(sid, bundle, verified):
    row = bundle['rows'][sid]
    path = row['target_path']
    if path not in verified:
        actual = sha256(path)
        if actual != row['target_sha256']:
            raise ValueError('Used raw teacher SHA mismatch: ' + sid)
        verified[path] = actual
    record = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
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


def io_stats():
    result = {}
    for line in Path('/proc/self/io').read_text().splitlines():
        key, value = line.split(':')
        result[key] = int(value)
    return result


def measured(operation):
    before = io_stats()
    result = operation()
    after = io_stats()
    return result, {key: after[key] - before[key] for key in ('rchar', 'read_bytes')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--training-out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    assert torch.serialization.get_default_mmap_options() == mmap.MAP_PRIVATE
    bundle = full.load_manifest(args.manifest, verify_sources=False)
    ids = [next(sid for sid in bundle['train'] if sid != 'FqlHJVDlFGLKCUJS'), 'FqlHJVDlFGLKCUJS', bundle['val'][0]]
    assert len(set(ids)) == 3
    old_verified, new_verified, samples = {}, {}, []
    report = dict(protocol='direct_teacher_readonly_mmap_actual3_cpu_v1', complete=False,
        pid=os.getpid(), hostname=os.uname().nodename, torch_version=torch.__version__,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), GPU_training_launched=False,
        old_sources_untouched=True, all6500_source_rehash=False, model_or_optimizer_loaded=False,
        mmap_options='MAP_PRIVATE_COW_no_inplace_write_file_unchanged', ids=ids, samples=samples)
    for sid in ids:
        row = bundle['rows'][sid]
        source = Path(row['target_path'])
        assert zipfile.is_zipfile(source), 'Legacy non-ZIP target is unsupported; no silent fallback'
        digest, hash_io = measured(lambda: sha256(source))
        assert digest == row['target_sha256']
        # Each reader retains its original once-per-source SHA check, independently.
        old, old_io = measured(lambda: original.raw_teacher(sid, bundle, old_verified))
        new, new_io = measured(lambda: raw_teacher(sid, bundle, new_verified))
        old_cached, old_cached_io = measured(lambda: original.raw_teacher(sid, bundle, old_verified))
        new_cached, new_cached_io = measured(lambda: raw_teacher(sid, bundle, new_verified))
        assert torch.equal(old, new) and torch.equal(old_cached, new_cached)
        with FakeTensorMode():
            meta = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
        tail_meta = meta['tail']
        storage_start = tail_meta.untyped_storage()._checkpoint_offset
        assert tail_meta.is_contiguous()
        tail_start = storage_start + tail_meta.storage_offset() * tail_meta.element_size()
        tensor_metadata = {}
        for key, value in meta.items():
            if isinstance(value, torch.Tensor):
                tensor_metadata[key] = dict(shape=list(value.shape), dtype=str(value.dtype),
                    storage_nbytes=value.untyped_storage().nbytes(),
                    storage_offset_elements=value.storage_offset(), stride=list(value.stride()),
                    storage_file_offset=value.untyped_storage()._checkpoint_offset)
            else:
                tensor_metadata[key] = dict(type=type(value).__name__, scalar=value if isinstance(value, (str,int,float,bool)) or value is None else None)
        with zipfile.ZipFile(source) as z:
            entries = [dict(name=v.filename, size=v.file_size, compressed_size=v.compress_size,
                compression=v.compress_type, local_header_offset=v.header_offset) for v in z.infolist()]
        sm, sr = _direct_sample_statistics(old[None], torch.ones(1,len(old),dtype=torch.bool))
        nm, nr = _direct_sample_statistics(new[None], torch.ones(1,len(new),dtype=torch.bool))
        assert torch.equal(sm,nm) and torch.equal(sr,nr)
        tail_sha = hashlib.sha256(old.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        sample = dict(sample_id=sid, path=str(source), file_bytes=source.stat().st_size,
            source_sha256=digest, tail_sha256=tail_sha, topkeys=list(meta), metadata=tensor_metadata,
            zip_entries=entries, tail_shape=list(new.shape), tail_dtype=str(new.dtype),
            tail_storage_offset_elements=tail_meta.storage_offset(),
            tail_file_window=[tail_start,tail_start+new.numel()*new.element_size()],
            tail_actual_bytes=new.numel()*new.element_size(),
            old_vs_mmap_bitwise_equal=True, mean_and_rms_bitwise_equal=True,
            original_manifest_moments_tolerance_passed=True,
            reads=dict(source_hash=hash_io,old_with_required_hash=old_io,mmap_with_required_hash=new_io,
                old_verified_cached=old_cached_io,mmap_verified_cached=new_cached_io))
        samples.append(sample)
        print(json.dumps(dict(event='actual_target_mmap_equal',sample=sample)),flush=True)
        del old,new,old_cached,new_cached,meta
    old_target, old_mask = original.teacher_batch(ids,bundle,torch.device('cpu'),old_verified)
    new_target, new_mask = teacher_batch(ids,bundle,torch.device('cpu'),new_verified)
    assert torch.equal(old_target,new_target) and torch.equal(old_mask,new_mask)
    tm,tr,groups = _direct_targets(ids,bundle,torch.device('cpu'))
    old_stats = _direct_sample_statistics(old_target,old_mask)
    new_stats = _direct_sample_statistics(new_target,new_mask)
    assert all(torch.equal(a,b) for a,b in zip(old_stats,new_stats))
    class CPUFixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Parameter(torch.randn(3,254,5120))
            self.adapter = torch.nn.Parameter(torch.zeros(5120))
        def forward(self):
            return self.backbone + self.adapter
    torch.manual_seed(42)
    random.seed(42)
    model = CPUFixture()
    def optimizer(current):
        return torch.optim.AdamW([dict(params=[current.backbone],lr=1e-5,name='backbone'),
            dict(params=[current.adapter],lr=1e-4,name='adapter')],weight_decay=.01)
    opt = optimizer(model)
    def step(current, current_opt, target, mask):
        current_opt.zero_grad(set_to_none=True)
        raw = current() + torch.randn_like(current.backbone) * (random.random()*.01)
        student_mask = torch.ones(3,254,dtype=torch.bool)
        loss,terms = original.objective('token',raw,student_mask,target,mask,tm,tr,
            bundle['mean'],bundle['std'],groups,1.)
        loss.backward()
        gradients = [p.grad.clone() for p in current.parameters()]
        norm = torch.nn.utils.clip_grad_norm_(current.parameters(),1.,error_if_nonfinite=True)
        current_opt.step()
        return loss.detach().clone(),{k:v.detach().clone() for k,v in terms.items()},gradients,norm
    step(model,opt,old_target,old_mask)
    resumed = copy.deepcopy(model)
    resumed_opt = optimizer(resumed)
    resumed_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    torch_rng, python_rng = torch.get_rng_state(), random.getstate()
    expected = step(model,opt,old_target,old_mask)
    torch.set_rng_state(torch_rng)
    random.setstate(python_rng)
    actual = step(resumed,resumed_opt,new_target,new_mask)
    assert torch.equal(expected[0],actual[0]) and torch.equal(expected[3],actual[3])
    assert all(torch.equal(expected[1][k],actual[1][k]) for k in expected[1])
    assert all(torch.equal(a,b) for a,b in zip(expected[2],actual[2]))
    assert all(torch.equal(value,resumed.state_dict()[key]) for key,value in model.state_dict().items())
    assert opt.state_dict()['param_groups'] == resumed_opt.state_dict()['param_groups']
    for key,state in opt.state_dict()['state'].items():
        assert all(torch.equal(value,resumed_opt.state_dict()['state'][key][name]) for name,value in state.items())
    checkpoint = {}
    for name in ['progress.json','latest.json','validation_epoch01.json','partial_failure.json','finished.json']:
        path = args.training_out/name
        if path.is_file():
            checkpoint[name] = json.loads(path.read_text())
    latest = checkpoint.get('latest.json',{})
    for key in ['checkpoint_name','train_state_name']:
        name = latest.get(key)
        if name:
            path = args.training_out/name
            checkpoint[name] = dict(exists=path.is_file(),bytes=path.stat().st_size if path.is_file() else None)
    report.update(complete=True,batch_padding_and_masks_bitwise_equal=True,
        CPU_next_step=dict(fixture_not_actual_MusicFM=True,actual_target_B=3,
            live_checkpoint_model_Adam_not_loaded=True,original_token_BxB_and_moments_weight=1.,
            loss=float(actual[0]),loss_terms_gradients_clip_parameters_Adam_bitwise_equal=True,
            RNG_restored=True),checkpoint_observation=checkpoint,
        source_fingerprints={str(Path(p).resolve()):sha256(p) for p in [__file__,original.__file__,full.__file__]},
        manifest_path=str(args.manifest),manifest_locked_sha256=original.MANIFEST_SHA,
        manifest_source_rehashed=False,target_source_hash_checks_retained=True)
    with args.out.open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(dict(event='mmap_cpu_complete',report=str(args.out),CPU_next_step=report['CPU_next_step'],checkpoint_observation=checkpoint)),flush=True)


if __name__ == '__main__':
    main()
