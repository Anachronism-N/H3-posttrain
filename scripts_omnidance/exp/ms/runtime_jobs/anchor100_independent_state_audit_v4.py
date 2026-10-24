"""Independent read-only anchor100 audit; never modifies original queue/state/source."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import numbers
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
MS = ROOT / 'diff/logs_omnidance/exp/MS'
RUN = '6452f0c2-0fbe-4f6e-b79a-01a169818bf5'
TRAIN = MS / ('h3_ms_style_anchor01_s100_n5_anchor_v8_' + RUN)
QUEUE = MS / ('h3_ms_style_train_queue_n5_anchor_v8_' + RUN)
MANAGER = MS / 'h3_ms_style_train_manager_n5_anchor_v8.py'
MANAGER_SHA = 'b7b3dc16943404867920b94cd62807b98a33f0d30f3b78e0ac776097771e905c'
SOURCE = ROOT / 'H3-posttrain/scripts_omnidance/exp/ms/h3_video_sft.py'
SOURCE_SHA = '247e77eb365685a92680bd3934ac172c7d81242f06fb28f7f4aa665130233c63'
WRAPPER_SHA = '89439d733d62f522bb3dd993bc456541b3ca15ead34ff2e19f86657a471c5055'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def numeric(value, expected):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value != expected:
        raise ValueError('Wrong or nonnumeric typed value: ' + repr(value))


def finite(value, label):
    import torch
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            flat = value.reshape(-1)
            for start in range(0, flat.numel(), 1048576):
                if not torch.isfinite(flat[start:start + 1048576]).all().item():
                    raise ValueError('Nonfinite tensor: ' + label)
    elif isinstance(value, dict):
        for key, child in value.items():
            finite(child, label + '/' + str(key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite(child, label + '/' + str(index))
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite scalar: ' + label)


def active_original():
    result = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            argv = [a.decode() for a in (p / 'cmdline').read_bytes().split(bytes([0])) if a]
            state = (p / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            continue
        if state == 'Z':
            continue
        prior_audit = str(MS / 'anchor100_independent_state_audit_v2.py') in argv
        if prior_audit or (str(TRAIN) in argv and str(SOURCE) in argv) or (str(MANAGER) in argv and RUN in argv):
            result.append(dict(pid=int(p.name), argv=argv, state=state))
    return result


def reconstruct(groups, offset, numel):
    import torch
    partition = math.ceil(numel / len(groups))
    parts = [group.narrow(0, offset, partition) for group in groups]
    return torch.cat(parts)[:numel], offset + partition


def self_test():
    import torch
    for value in (True, False, '100', float('nan'), float('inf')):
        try:
            numeric(value, 100)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid numeric accepted')
    numeric(100, 100)
    import numpy as np
    numeric(np.int64(800), 800)
    try:
        numeric(np.bool_(True), 1)
    except ValueError:
        pass
    else:
        raise AssertionError('NumPy boolean accepted')
    numeric(.1, .1)
    finite({'x': torch.tensor([1., 2.])}, 'ok')
    for bad in (torch.tensor([float('nan')]), {'x': float('inf')}):
        try:
            finite(bad, 'bad')
        except ValueError:
            pass
        else:
            raise AssertionError('Nonfinite accepted')
    parts = [torch.tensor([float(i), 99., float(i + 10)]) for i in range(8)]
    first, offset = reconstruct(parts, 0, 5)
    second, final = reconstruct(parts, offset + 1, 8)
    assert torch.equal(first, torch.arange(5).float()) and offset == 1
    assert torch.equal(second, torch.arange(10, 18).float()) and final == 3
    print(json.dumps(dict(complete=True, synthetic_only=True, typed_numeric_nonfinite_and_zero3_padding_tests=True)))


def audit(out):
    import numpy as np
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2)
    assert not active_original(), 'Original producer/manager still live; refuse competing full-state scan'
    assert sha(MANAGER) == MANAGER_SHA and sha(SOURCE) == SOURCE_SHA
    assert sha(SOURCE.with_name('h3_video_only.py')) == WRAPPER_SHA
    state = TRAIN / 'training-state-final-100'
    marker = json.loads((state / 'complete.json').read_text())
    sig, progress = marker['signature'], marker['progress']
    assert marker['scope'] == sig['data_scope'] == 'source_label_train'
    assert progress['signature'] == sig
    for key, expected in {'optimizer_updates': 100, 'microbatches': 100}.items():
        assert type(progress[key]) is int
        numeric(progress[key], expected)
    for key, expected in {'anchor_lambda': .1, 'world_size': 8, 'seed': 42,
                          'lora_rank': 32, 'lora_alpha': 64, 'learning_rate': 1e-5,
                          'accumulation': 1, 'dataset_size': 512}.items():
        numeric(sig[key], expected)
    assert sig['sources']['code'] == {SOURCE.name: SOURCE_SHA, 'h3_video_only.py': WRAPPER_SHA}
    assert len(marker['files']) == 28
    assert all(type(record['bytes']) is int and record['bytes'] > 0 for record in marker['files'].values())
    args = json.loads((TRAIN / 'training_args.json').read_text())
    for key, expected in {'max_train_steps': 100, 'training_cfg_scale': 1., 'anchor_lambda': .1,
                          'height': 1280, 'width': 704, 'num_frames': 124}.items():
        numeric(args[key], expected)
    spec = importlib.util.spec_from_file_location('original_anchor_manager_audit_only', MANAGER)
    manager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manager)
    sys.path.insert(0, str(SOURCE.parent))
    import h3_video_sft as h3
    save(out / 'hash_scan_started.json', dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         original_verifier=str(MANAGER), original_verifier_sha256=MANAGER_SHA, members=28))
    receipt = manager.verify_training(h3, TRAIN, .1)
    save(out / 'original_verifier_passed.json', receipt)
    custom = torch.load(state / 'custom_checkpoint_0.pkl', map_location='cpu', weights_only=False)
    assert custom == progress
    scheduler = torch.load(state / 'scheduler.bin', map_location='cpu', weights_only=False)
    numeric(scheduler['last_epoch'], progress['optimizer_updates'] * sig['world_size'])
    numeric(scheduler['_step_count'], scheduler['last_epoch'] + 1)
    assert type(scheduler['base_lrs']) is list and type(scheduler['_last_lr']) is list
    assert len(scheduler['base_lrs']) == len(scheduler['_last_lr']) == 1
    numeric(scheduler['base_lrs'][0], 1e-5)
    numeric(scheduler['_last_lr'][0], 1e-5)
    finite(scheduler, 'scheduler')
    shapes = None
    for name in receipt['model_shards']:
        model = torch.load(state / name, map_location='cpu', weights_only=False, mmap=True)
        for key, expected in {'global_steps': 100, 'global_samples': 800, 'dp_world_size': 8,
                              'mp_world_size': 1, 'skipped_steps': 0}.items():
            numeric(model[key], expected)
        assert model['lr_scheduler'] is None, 'This recipe stores scheduler separately in Accelerate'
        if shapes is None:
            shapes = model['param_shapes']
        assert model['param_shapes'] == shapes
        assert model['module'] and model['frozen_param_fragments']
        finite(model['module'], name + '/module')
        finite(model['frozen_param_fragments'], name + '/frozen')
        save(out / (Path(name).name + '.verified.json'), dict(complete=True, finite=True))
        del model
    masters = []
    for name in receipt['optimizer_shards']:
        value = torch.load(state / name, map_location='cpu', weights_only=False, mmap=True)
        optimizer = value['optimizer_state_dict']
        numeric(optimizer['zero_stage'], 3)
        numeric(optimizer['partition_count'], 8)
        assert optimizer['overflow'] is False
        adam = optimizer['optimizer_state_dict']
        assert adam['state'] and len(optimizer['fp32_flat_groups']) == len(shapes)
        for av in adam['state'].values():
            assert isinstance(av['step'], torch.Tensor) and av['step'].numel() == 1
            numeric(av['step'].item(), 100)
            assert av['exp_avg'].shape == av['exp_avg_sq'].shape and av['exp_avg'].numel() > 0
        for group in adam['param_groups']:
            numeric(group['lr'], 1e-5)
        finite(adam, name + '/adam')
        finite(optimizer['fp32_flat_groups'], name + '/master')
        masters.append(optimizer['fp32_flat_groups'])
        save(out / (Path(name).name + '.verified.json'), dict(complete=True, finite=True, actual_adam_step=100))
    for rank in range(8):
        rng = torch.load(state / ('random_states_' + str(rank) + '.pkl'), map_location='cpu', weights_only=False)
        random.Random().setstate(rng['random_state'])
        np.random.RandomState().set_state(rng['numpy_random_seed'])
        assert rng['torch_manual_seed'].dtype == torch.uint8 and rng['torch_manual_seed'].numel() == 5056
        torch.Generator(device='cpu').set_state(rng['torch_manual_seed'])
        assert len(rng['torch_cuda_manual_seed']) == 8
        assert all(isinstance(v, torch.Tensor) and v.dtype == torch.uint8 and v.numel() > 0 for v in rng['torch_cuda_manual_seed'])
        finite(rng, 'rng_' + str(rank))
    lora = TRAIN / 'step-100.safetensors'
    matched = []
    with safe_open(str(lora), framework='pt', device='cpu') as stream:
        keys = set(stream.keys())
        for group_index, group_shapes in enumerate(shapes):
            groups = [rank[group_index] for rank in masters]
            offset = 0
            for name, shape in group_shapes.items():
                assert name.startswith('pipe.dit.base_model.model.') and ('.lora_A.' in name or '.lora_B.' in name)
                key = name.removeprefix('pipe.dit.base_model.model.')
                assert key in keys
                exported = stream.get_tensor(key)
                assert list(exported.shape) == list(shape) and exported.dtype == torch.bfloat16
                assert exported.shape[0 if '.lora_A.' in key else 1] == 32
                finite(exported, key)
                master, offset = reconstruct(groups, offset, math.prod(shape))
                expected = master.reshape(shape).to(exported.dtype)
                if '.lora_B.' in key:
                    expected = expected * 2.
                assert torch.equal(exported, expected), 'LoRA export differs from checkpoint master: ' + key
                counterpart = key.replace('.lora_A.', '.lora_B.') if '.lora_A.' in key else key.replace('.lora_B.', '.lora_A.')
                assert counterpart in keys
                matched.append(key)
            assert all(offset == group.numel() for group in groups)
        assert set(matched) == keys and len(matched) == len(keys), 'All and only trainable A/B must match'
    original_terminal = {p.name: json.loads(p.read_text()) for p in [QUEUE / 'anchor01_exit.json', QUEUE / 'finished.json'] if p.is_file()}
    result = dict(receipt, full_checkpoint_reverified=True, independent_audit=True,
         actual_training_updates=100, world_size=8, lora_rank=32, lora_alpha=64, learning_rate=1e-5,
         anchor_lambda=.1, training_cfg_scale=1., fm_recipe_verified=True, typed_numeric_verified=True,
         model_and_adam_and_rng_and_scheduler_finite=True, complete_state_members=28,
         accelerate_scheduler_last_epoch=scheduler['last_epoch'],
         accelerate_scheduler_step_count=scheduler['_step_count'],
         scheduler_count_policy='Accelerate world-size-scaled scheduling: 100 optimizer updates * world8 = 800',
         all_file_sha256_verified=True, scheduler_storage='Accelerate scheduler.bin; DeepSpeed lr_scheduler=None',
         source_sha256=SOURCE_SHA, wrapper_sha256=WRAPPER_SHA, original_verifier_sha256=MANAGER_SHA,
         lora=str(lora), lora_sha256=sha(lora), lora_tensor_keys=len(matched),
         lora_all_ab_source_verified=True, lora_origin='all ZeRO3 fp32 master partitions -> BF16; B scaled alpha/r=2 once',
         original_queue_terminal=original_terminal, original_exit_code_preserved=True,
         original_queue_modified=False, retraining_started=False, gpu_forward_performed=False,
         human_quality_verified=False, utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(out / 'full_state_verified.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--wait-seconds', type=int, default=86400)
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    assert args.out and not args.out.exists() and os.getuid() == 0
    args.out.mkdir()
    save(args.out / 'audit_live.json', dict(pid=os.getpid(), source_sha256=sha(__file__), argv=sys.argv, gpu_forward_performed=False))
    deadline = time.monotonic() + args.wait_seconds
    try:
        while True:
            active = active_original()
            ready = (TRAIN / 'training-state-final-100/complete.json').is_file()
            if not active and ready:
                break
            if not active and not ready:
                raise RuntimeError('Producer exited without final marker; preserve assets')
            if time.monotonic() >= deadline:
                raise RuntimeError('Original producer/manager wait exhausted; no duplicate/retry')
            with (args.out / 'wait_events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    active_original=active, final_marker_present=ready, gpu_forward_performed=False)) + '\n')
            time.sleep(60)
        result = audit(args.out)
        save(args.out / 'finished.json', dict(complete=True, full_checkpoint_reverified=True,
            report=str(args.out / 'full_state_verified.json'), lora_sha256=result['lora_sha256']))
    except Exception as error:
        save(args.out / 'finished.json', dict(complete=False, error=repr(error), original_assets_preserved=True, retry_started=False))
        raise


if __name__ == '__main__':
    main()
