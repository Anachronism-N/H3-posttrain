"""Independently reload scale checkpoints; publish metadata only, never model tensors."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from music_clip_bridge import INTERFACE, ids_fingerprint
from music_generation_validate import atomic_write_new


def audit(root, names):
    hashes, results, common = {}, [], None

    def digest(path):
        path = str(path)
        if path not in hashes:
            with open(path, 'rb') as stream:
                hashes[path] = hashlib.file_digest(stream, 'sha256').hexdigest()
        return hashes[path]

    for name in names:
        directory = root / name
        summary_path = directory / 'summary.json'
        summary = json.loads(summary_path.read_text())
        config = summary['config']
        bridge = config['stage'] == 'bridge'
        assert config['backend'] == 'musicfm'
        assert config['stage'] in ('bridge', 'bridge_clip')
        assert config['max_steps'] in ((100,) if bridge else (560, 1875))
        assert config['eval_every_steps'] == config['max_steps']
        assert summary['global_step'] == summary['target_steps'] == config['max_steps']
        assert summary['world_size'] == 8
        assert summary['global_batch'] == (256 if bridge else 16)
        assert summary['actually_items_seen'] == summary['global_step'] * summary['global_batch']
        assert config['encoder_lr'] == 1e-5 and config['head_lr'] == 1e-4
        assert config['adapter_lr'] == (3e-4 if bridge else 1e-4)
        assert config['bridge_weight'] in (0., 1.)
        assert bool(config['features']) == bridge
        assert bool(config['init']) != bridge
        assert config['teacher'] and not config['allow_init_train_mismatch']
        split = json.loads(Path(config['split']).read_text())
        normalization_ids = json.loads(Path(config['normalization_ids']).read_text())
        assert summary['split'] == split
        assert len(split['train']) in (1800, 6000) and len(split['val']) == 200
        assert len(set(split['train'])) == len(split['train'])
        assert len(set(split['val'])) == len(split['val'])
        assert not set(split['train']) & set(split['val'])
        assert len(normalization_ids) == len(set(normalization_ids)) == 1800
        assert set(normalization_ids) <= set(split['train'])
        assert summary['normalization']['ids'] == normalization_ids
        assert summary['normalization']['ids_sha256'] == ids_fingerprint(normalization_ids)
        for key, claimed in summary['input_sha256'].items():
            assert digest(config[key]) == claimed, (name, key)
        counts = summary['gradients']
        for group, expected in [('backbone', 0 if bridge else 316343296),
                                ('adapter', 469536), ('audio_head', 0 if bridge else 655616)]:
            assert counts[group]['parameters_with_grad'] == expected, (name, group)
        assert counts['backbone']['missing_grad_tensors'] == (0 if bridge else 3)
        key = (split['val'], normalization_ids, summary['normalization']['stats_sha256'],
               summary['input_sha256']['caps'], summary['input_sha256']['text'],
               summary['input_sha256']['teacher'])
        if common is None:
            common = key
        assert key == common, f'{name}: fixed val/norm/source hashes changed'
        checkpoints = []
        for label in ('best', 'last'):
            path = directory / (label + '.pt')
            actual_hash = digest(path)
            assert actual_hash == summary['checkpoint_sha256'][label], (name, label)
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            for field in ('config', 'split', 'input_sha256', 'normalization', 'gradients'):
                assert checkpoint[field] == summary[field], (name, label, field)
            assert checkpoint['interface'] == INTERFACE
            assert checkpoint['stage'] == config['stage']
            assert checkpoint['global_step'] == checkpoint['target_steps'] == config['max_steps']
            assert checkpoint['world_size'] == 8
            assert checkpoint['global_batch'] == summary['global_batch']
            assert checkpoint['metrics']['val_candidates'] == 200
            if label == 'best':
                assert checkpoint['metrics'] == summary['best_metrics']
            groups = ('adapter',) if bridge else ('adapter', 'model')
            tensor_count, floating_values = 0, 0
            for group in groups:
                assert checkpoint[group]
                for tensor in checkpoint[group].values():
                    assert isinstance(tensor, torch.Tensor)
                    assert torch.isfinite(tensor).all(), (name, label, group)
                    tensor_count += 1
                    floating_values += tensor.numel() if tensor.is_floating_point() else 0
            mean, std = checkpoint['text_mean'], checkpoint['text_std']
            assert mean.shape == std.shape == (5120,)
            assert torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()
            stats_hash = hashlib.sha256(mean.contiguous().numpy().tobytes() +
                                        std.contiguous().numpy().tobytes()).hexdigest()
            assert stats_hash == summary['normalization']['stats_sha256']
            checkpoints.append(dict(name=label, sha256=actual_hash, tensors_finite=True,
                                    tensor_count=tensor_count, floating_values=floating_values,
                                    global_step=checkpoint['global_step'], metrics=checkpoint['metrics']))
            del checkpoint
        events = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
        assert events[-1]['event'] == 'finished'
        assert events[-1]['global_step'] == config['max_steps']
        assert any(e['event'] == 'validation' and e['reason'] == 'final' and
                   e['global_step'] == config['max_steps'] for e in events)
        result = dict(name=name, status='verified', stage=config['stage'],
                      train_count=len(split['train']), val_count=200,
                      train_ids_sha256=ids_fingerprint(split['train']),
                      val_ids_sha256=ids_fingerprint(split['val']),
                      normalization_ids_sha256=ids_fingerprint(normalization_ids),
                      normalization_stats_sha256=summary['normalization']['stats_sha256'],
                      global_step=summary['global_step'], global_batch=summary['global_batch'],
                      actually_items_seen=summary['actually_items_seen'],
                      bridge_weight=config['bridge_weight'], encoder_lr=config['encoder_lr'],
                      adapter_lr=config['adapter_lr'], head_lr=config['head_lr'],
                      input_sha256=summary['input_sha256'], summary_sha256=digest(summary_path),
                      checkpoints=checkpoints, best_metrics=summary['best_metrics'],
                      gradients=counts, performance=summary['performance'],
                      bn_caveat=summary['bn_caveat'])
        results.append(result)
        print(json.dumps(dict(name=name, status='verified', global_step=result['global_step']),
                         allow_nan=False), flush=True)
    return dict(status='verified', configuration_count=len(results),
                checkpoint_count=2 * len(results), configurations=results,
                fixed_validation_and_normalization_verified=True,
                scope='Checkpoint and training-protocol integrity only; not generated video acceptance or musical adherence.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dirs', nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.root.is_dir() or not args.out.parent.is_dir() or args.out.exists():
        parser.error('Require existing input/output parents and a new report path')
    if len(set(args.dirs)) != len(args.dirs) or any(Path(n).name != n or n in ('.', '..') for n in args.dirs):
        parser.error('Require unique bare directory names')
    torch.set_num_threads(2)
    result = audit(args.root, args.dirs)
    atomic_write_new(args.out, result)
    assert json.loads(args.out.read_text()) == result
    print(json.dumps(dict(status=result['status'], checkpoints=result['checkpoint_count']), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
