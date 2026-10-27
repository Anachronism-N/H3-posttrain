"""Source-balanced native-audio planner SFT with optional input-only JEV genre loss.

An explicit contract enables new captions; otherwise historical fixed2s data is
preserved. Test data is never consumed by this training entry.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import tempfile
import uuid

PROCESS_IDENTITY = dict(host=socket.gethostname(), pid=os.getpid(), nonce=uuid.uuid4().hex)

import torch
import torch.nn.functional as F

SCHEMA = 'music-planner-train-v1'
FAMILIES = ('moss', 'music_flamingo', 'omni_captioner')
LENGTH_POLICY = 'one-full-5.0625s-audio-two-text-windows-no-truncation-v1'
ZERO3_PARAMETER_LIFECYCLE = 'native-partition-all-then-reset-after-validation-before-save-v1'


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False, encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def read_split(path, split):
    if split not in ('train', 'val'):
        raise ValueError('Training entry accepts train/val only; test is final-only')
    path = Path(path).resolve()
    rows, ids, groups, audios = [], set(), set(), set()
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('split') != split:
                raise ValueError(f'{path}:{number}: expected {split}, test/mixed splits forbidden')
            for key in ('id', 'source_group_id', 'audio_path', 'audio_sha256', 'target_text', 'caption_version'):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f'{path}:{number}: missing {key}')
            if row['id'] in ids:
                raise ValueError('Each source id must occur once per epoch, not one example per 2s window')
            if len(row['audio_sha256']) != 64 or any(c not in '0123456789abcdef' for c in row['audio_sha256']):
                raise ValueError('Invalid declared audio SHA256')
            if row.get('audio_window_seconds') != [0.0, 5.0625]:
                raise ValueError('Require original full [0,5.0625] audio, never physical 2s crops')
            windows = row.get('target_windows', row.get('target_windows2'))
            if not isinstance(windows, list) or len(windows) != 2:
                raise ValueError('Require two target windows in one source example')
            if [(w.get('start'), w.get('end')) for w in windows] != [(0.0, 2.0), (2.0, 5.0625)]:
                raise ValueError('Fixed2s target timing changed')
            if any(not isinstance(w.get('caption'), str) or not w['caption'].strip() for w in windows):
                raise ValueError('Empty target window caption')
            if any(row.get(k) is not False for k in ('semantic_approval', 'human_review_completed')):
                raise ValueError('Weak labels must not claim semantic/human approval')
            if '<|' in row['target_text'] or '<audio>' in row['target_text']:
                raise ValueError('Reserved chat/audio token in target')
            audio = Path(row['audio_path']).expanduser()
            if not audio.is_absolute():
                audio = path.parent / audio
            row = dict(row, audio_path=os.path.abspath(audio))
            ids.add(row['id'])
            groups.add(row['source_group_id'])
            audios.add(row['audio_sha256'])
            rows.append(row)
    if not rows:
        raise ValueError(f'Empty {split} split')
    return rows, dict(path=str(path), sha256=file_sha(path), sources=len(rows), isolation_groups=len(groups),
                     source_groups_sha256=object_sha(sorted(groups)), audio_identities_sha256=object_sha(sorted(audios)))


def check_data(train_path, val_path):
    train, train_id = read_split(train_path, 'train')
    val, val_id = read_split(val_path, 'val')
    for key in ('id', 'source_group_id', 'audio_sha256', 'audio_path'):
        if {r[key] for r in train} & {r[key] for r in val}:
            raise ValueError(f'Train/val leakage: {key}')
    return train, val, dict(train=train_id, val=val_id, length_policy=LENGTH_POLICY,
        source_overlap=0, exact_audio_overlap=0, song_near_duplicate_status='unknown',
        semantic_approval=False, human_review_completed=False, test_used=False)


class AudioIdentityCache:
    """Hash each consumed file once; reuse only while its stat identity is unchanged."""
    def __init__(self, path):
        self.path = Path(path)
        self.entries = json.loads(self.path.read_text(encoding='utf-8')) if self.path.is_file() else {}
        self.dirty = False

    def verify(self, row):
        path = Path(row['audio_path'])
        stat = path.stat()
        key = str(path.resolve())
        identity = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=row['audio_sha256'])
        if self.entries.get(key) == identity:
            return
        if file_sha(path) != row['audio_sha256']:
            raise ValueError(f'Audio bytes changed: {row["id"]}')
        self.entries[key] = identity
        self.dirty = True

    def flush(self):
        if self.dirty:
            atomic_json(self.path, self.entries)
            self.dirty = False


def epoch_order(size, seed, epoch):
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    return order


def update_slices(order, position, micro, accum, rank, world):
    selected = order[position:position + micro * accum * world]
    result = []
    for step in range(accum):
        start = (step * world + rank) * micro
        result.append(selected[start:start + micro])
    return result, len(selected)


def validate_cursor(state, size, epochs, global_batch):
    epoch, position, update = state['epoch'], state['position'], state['update']
    if any(type(v) is not int or v < 0 for v in (epoch, position, update, state['source_exposures'])):
        raise ValueError('Invalid resume cursor')
    if epoch > epochs or position >= size or (epoch == epochs and position != 0):
        raise ValueError('Resume epoch/position out of bounds')
    if position % global_batch:
        raise ValueError('Checkpoint is not at an optimizer boundary')
    expected_updates = epoch * math.ceil(size / global_batch) + position // global_batch
    if update != expected_updates or state['source_exposures'] != epoch * size + position:
        raise ValueError('Resume update and exact source exposures disagree')


def planner_module(name):
    # Direct script execution and package execution resolve the same repository code.
    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module('scripts_omnidance.exp.ms.' + name)


def shifted_nll(logits, labels, chunk_size=0):
    return planner_module('music_planner_token_loss').assistant_token_nll(logits, labels, chunk_size)


def cluster_environment(world):
    plan = os.environ.get('PLANNER_CLUSTER_PLAN_SHA256')
    if plan is None:
        return None
    if len(plan) != 64 or any(c not in '0123456789abcdef' for c in plan):
        raise ValueError('Invalid cluster plan SHA256')
    if (int(os.environ.get('PLANNER_EXPECTED_WORLD_SIZE', '0')) != world or
            int(os.environ.get('LOCAL_WORLD_SIZE', '0')) != 8 or world not in (32, 64) or
            os.environ.get('PLANNER_EXPECTED_HOSTNAME') != socket.gethostname()):
        raise ValueError('Cluster plan world/local-world/host differs from this process')
    inventory = os.environ.get('PLANNER_BASE_MODEL_INVENTORY_SHA256', '')
    if len(inventory) != 64 or any(c not in '0123456789abcdef' for c in inventory):
        raise ValueError('Cluster execution requires the content-bound base model inventory SHA256')
    return dict(plan_sha256=plan, world_size=world, processes_per_node=8,
                base_model_inventory_sha256=inventory)


def canonical_cluster_model_identity(identity, leader_records, expected_sha, expected_nodes):
    if (len(leader_records) != expected_nodes or any(record.get('error') for record in leader_records) or
            any(record.get('inventory_sha256') != expected_sha for record in leader_records)):
        raise ValueError('Node-leader base model content verification failed or differs from cluster plan: ' +
                         json.dumps(leader_records, ensure_ascii=False))
    result = dict(identity)
    result['weights'] = [{key: value for key, value in weight.items() if key != 'mtime_ns'}
                         for weight in identity['weights']]
    result.update(identity_method='actual-full-directory-content-inventory-verified-on-every-node-v1',
                  content_inventory_sha256=expected_sha, content_verified_nodes=expected_nodes)
    return result


def verify_cluster_model_identity(model_path, identity, accelerator, cluster):
    if cluster is None:
        return identity
    record = None
    if accelerator.local_process_index == 0:
        try:
            inventory = planner_module('music_planner_cluster').model_inventory(model_path)
            record = dict(inventory_sha256=inventory['inventory_sha256'], file_count=len(inventory['files']))
        except Exception as error:
            # All ranks reach the collective even if one node cannot read its weights.
            record = dict(error=type(error).__name__ + ': ' + str(error))
    records = [None] * accelerator.num_processes
    torch.distributed.all_gather_object(records, record)
    leaders = [value for value in records if value is not None]
    return canonical_cluster_model_identity(identity, leaders, cluster['base_model_inventory_sha256'],
                                            cluster['world_size'] // cluster['processes_per_node'])


def configure_zero3_leaf_modules(model, family, zero3):
    if not zero3 or family != 'omni_captioner':
        return []
    blocks = [(name, module) for name, module in model.named_modules()
              if type(module).__name__ == 'Qwen3OmniMoeThinkerTextSparseMoeBlock']
    if not blocks:
        raise ValueError('Native Captioner MoE blocks not found; cannot verify ZeRO3 leaf policy')
    from deepspeed.utils import set_z3_leaf_modules
    set_z3_leaf_modules(model, list({type(module) for _, module in blocks}))
    if any(not getattr(module, '_z3_leaf', False) for _, module in blocks):
        raise ValueError('DeepSpeed did not apply the requested MoE leaf policy')
    return [dict(name=name, class_name=type(module).__name__) for name, module in blocks]


def lr_multiplier(step, total_updates, warmup_updates, schedule='constant'):
    if schedule not in ('constant', 'cosine'):
        raise ValueError('Unknown learning rate schedule')
    if step < warmup_updates or schedule == 'constant':
        return min(1., (step + 1) / max(1, warmup_updates))
    progress = min(1., max(0., (step - warmup_updates + 1) / max(1, total_updates - warmup_updates)))
    return .5 * (1. + math.cos(math.pi * progress))


def assert_shared_fingerprint(fingerprint):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        observed = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(observed, fingerprint)
        if len(set(observed)) != 1:
            raise ValueError('Distributed ranks disagree on model/data/code/training configuration')


def software_identity():
    versions = {}
    for name in ('torch', 'accelerate', 'deepspeed', 'transformers', 'peft', 'numpy', 'scipy', 'soundfile'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=sys.version.split()[0], packages=versions)


def prepare_training_storage(model, loader, zero3):
    base_dtype = torch.float32 if zero3 else torch.bfloat16
    if zero3:
        cast_parameter_dtype = importlib.import_module(loader.__module__)._cast_parameter_dtype
        for parameter in model.parameters():
            if parameter.is_floating_point():
                cast_parameter_dtype(parameter, torch.float32)
    return base_dtype


def optimizer_groups(model, lora_lr, projector_lr, base_dtype=torch.bfloat16, genre_lr=1e-4):
    groups = {'lora': [], 'projector': [], 'jev': []}
    names = {'lora': [], 'projector': [], 'jev': []}
    for name, parameter in model.named_parameters():
        expected_dtype = torch.float32 if parameter.requires_grad else base_dtype
        for attribute in ('ds_tensor', 'ds_secondary_tensor'):
            partition = getattr(parameter, attribute, None)
            if partition is not None and partition.is_floating_point() and partition.dtype != expected_dtype:
                raise ValueError(f'ZeRO partition dtype disagrees with parameter contract: {name}: {attribute}: {partition.dtype}')
        if not parameter.requires_grad:
            if parameter.is_floating_point() and parameter.dtype != base_dtype:
                raise ValueError(f'Frozen base must be {base_dtype}: {name}: {parameter.dtype}')
            continue
        if parameter.dtype != torch.float32:
            raise ValueError(f'Trainable parameter must stay FP32: {name}: {parameter.dtype}')
        if any(token in name.lower() for token in ('visual', 'vision_tower', 'talker', 'code2wav', 'lm_head')):
            raise ValueError(f'Forbidden music-only trainable parameter: {name}')
        kind = 'jev' if 'jev_head' in name.split('.') else 'lora' if 'lora_' in name else 'projector'
        if kind == 'projector' and not any(token in name for token in ('projector', 'projection', 'connector', '.proj', '.modules_to_save.')):
            raise ValueError(f'Unexpected unfrozen base parameter: {name}')
        groups[kind].append(parameter)
        names[kind].append(name)
    if not groups['lora']:
        raise ValueError('Adapter has no trainable LoRA parameters')
    result = []
    for kind, lr in (('lora', lora_lr), ('projector', projector_lr), ('jev', genre_lr)):
        if groups[kind]:
            result.append(dict(params=groups[kind], lr=lr, name=kind))
    if not names['jev']:
        # Preserve the historical group signature for caption-only checkpoints.
        names.pop('jev')
    return result, names


def parameter_signature(model):
    return {name: dict(shape=list(getattr(p, 'ds_shape', p.shape)), dtype=str(p.dtype))
            for name, p in model.named_parameters() if p.requires_grad}


def capture_rng():
    result = dict(python=random.getstate(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        result['cuda'] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
        result['numpy'] = np.random.get_state()
    except ImportError:
        pass
    return result


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])
    if 'numpy' in state:
        import numpy as np
        np.random.set_state(state['numpy'])


def finish_zero3_forward(model, zero3=False):
    if not zero3:
        return
    # The final no-grad forward has no following forward hook/optimizer step to drain prefetches.
    offload = model.optimizer.parameter_offload
    offload.partition_all_parameters()
    offload.get_param_coordinator().reset_step()


def trainable_state(model, rank=0, zero3=False):
    result = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if zero3:
            import deepspeed
            context = deepspeed.zero.GatheredParameters([parameter], modifier_rank=0)
        else:
            context = nullcontext()
        with context:
            if rank == 0:
                result[name] = parameter.detach().cpu().clone()
    return result


def restore_trainable(model, saved, zero3=False):
    signature = parameter_signature(model)
    if set(saved) != set(signature):
        raise ValueError('Strict trainable state keys differ')
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        value = saved[name]
        if list(value.shape) != signature[name]['shape'] or str(value.dtype) != signature[name]['dtype']:
            raise ValueError(f'Strict trainable state shape/dtype differs: {name}')
        if zero3:
            import deepspeed
            context = deepspeed.zero.GatheredParameters([parameter], modifier_rank=0)
        else:
            context = nullcontext()
        with context, torch.no_grad():
            parameter.copy_(value.to(parameter.device))


def file_ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path), size_bytes=path.stat().st_size)


def optimizer_evidence(optimizer):
    # ZeRO3's public state_dict restores its flattened FP32 parameter groups before serializing Adam.
    full_state = optimizer.state_dict()
    if 'optimizer_state_dict' in full_state:
        full_state = full_state['optimizer_state_dict']
    state = full_state['state']
    moments = [value[key] for value in state.values() for key in ('exp_avg', 'exp_avg_sq') if key in value]
    digest = hashlib.sha256(json.dumps(full_state['param_groups'], sort_keys=True, default=str).encode())
    for index, value in sorted(state.items()):
        digest.update(str(index).encode())
        for key, item in sorted(value.items()):
            digest.update(key.encode())
            if isinstance(item, torch.Tensor):
                cpu = item.detach().cpu().contiguous()
                digest.update(str((cpu.dtype, tuple(cpu.shape))).encode())
                digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
            else:
                digest.update(repr(item).encode())
    return dict(optimizer_state_sha256=digest.hexdigest(), optimizer_state_initialized=bool(moments), moment_tensors=len(moments),
                moments_fp32=bool(moments) and all(v.dtype == torch.float32 for v in moments),
                moments_finite=bool(moments) and all(bool(torch.isfinite(v).all()) for v in moments))


def assert_rng_equal(expected, actual):
    if expected['python'] != actual['python'] or not torch.equal(expected['torch'], actual['torch']):
        raise ValueError('Strict Python/Torch RNG restoration differs')
    if 'cuda' in expected and not all(torch.equal(a, b) for a, b in zip(expected['cuda'], actual['cuda'])):
        raise ValueError('Strict CUDA RNG restoration differs')
    if 'numpy' in expected:
        import numpy as np
        a, b = expected['numpy'], actual['numpy']
        if a[0] != b[0] or not np.array_equal(a[1], b[1]) or a[2:] != b[2:]:
            raise ValueError('Strict NumPy RNG restoration differs')


def assert_portable_equal(expected, actual):
    if set(expected) != set(actual):
        raise ValueError('Strict restored trainable keys differ')
    for name in expected:
        if expected[name].dtype != actual[name].dtype or not torch.equal(expected[name], actual[name]):
            raise ValueError(f'Strict DS restored trainable bytes differ: {name}')


def save_checkpoint(path, model, optimizer, scheduler, state, fingerprint, rank=0, world=1,
                    zero3=False, barrier=lambda: None, adapter=None):
    path = Path(path)
    finish_zero3_forward(model, zero3)
    if rank == 0:
        path.mkdir(parents=True, exist_ok=False)
    barrier()
    unwrapped = model.module if hasattr(model, 'module') else model
    portable = trainable_state(unwrapped, rank, zero3)
    if rank == 0:
        torch.save(portable, path / 'trainable.pt')
        torch.save(scheduler.state_dict(), path / 'scheduler.pt')
        if adapter is not None:
            adapter.save_pretrained(path / 'adapter', state_dict=portable)
            if hasattr(unwrapped, 'jev_head'):
                planner_module('music_planner_jev').export_jev(
                    adapter, path / 'jev', identity=dict(run_fingerprint=fingerprint,
                        trainable_sha256=file_sha(path / 'trainable.pt'),
                        installation=getattr(adapter, 'jev_metadata', None)), state_dict=portable)
    if zero3:
        model.save_checkpoint(str(path / 'distributed'), tag='zero3', exclude_frozen_parameters=True)
    else:
        torch.save(optimizer.state_dict(), path / f'optimizer-rank{rank}.pt')
    torch.save(capture_rng(), path / f'rng-rank{rank}.pt')
    atomic_json(path / f'writer-rank{rank}.json', dict(PROCESS_IDENTITY, rank=rank,
        optimizer=optimizer_evidence(optimizer)))
    barrier()
    if rank == 0:
        atomic_json(path / 'trainer_state.json', dict(schema=SCHEMA, fingerprint=fingerprint,
            world_size=world, zero3=zero3, trainable_signature=parameter_signature(unwrapped),
            cursor=state, trainable_sha256=file_sha(path / 'trainable.pt'),
            files=[file_ref(p) for p in sorted(path.rglob('*')) if p.is_file()], complete=True))
    barrier()
    return path


def load_checkpoint(path, model, optimizer, scheduler, fingerprint, rank=0, world=1,
                    zero3=False, barrier=lambda: None):
    path = Path(path)
    metadata = json.loads((path / 'trainer_state.json').read_text(encoding='utf-8'))
    unwrapped = model.module if hasattr(model, 'module') else model
    if (metadata.get('schema') != SCHEMA or metadata.get('complete') is not True or
        metadata.get('fingerprint') != fingerprint or metadata.get('world_size') != world or
        metadata.get('zero3') != zero3 or metadata.get('trainable_signature') != parameter_signature(unwrapped)):
        raise ValueError('Strict checkpoint identity/config/world/trainable signature mismatch')
    if file_sha(path / 'trainable.pt') != metadata['trainable_sha256']:
        raise ValueError('Checkpoint trainable bytes changed')
    if rank == 0:
        for ref in metadata.get('files', []):
            if file_sha(ref['path']) != ref['sha256']:
                raise ValueError(f'Checkpoint file bytes changed: {ref["path"]}')
    barrier()
    writer = json.loads((path / f'writer-rank{rank}.json').read_text(encoding='utf-8'))
    if zero3 and writer['nonce'] == PROCESS_IDENTITY['nonce']:
        raise ValueError('ZeRO3 resume requires a fresh process/engine, never same-engine reload')
    saved = torch.load(path / 'trainable.pt', map_location='cpu', weights_only=True)
    if zero3:
        loaded, _ = model.load_checkpoint(str(path / 'distributed'), tag='zero3',
            load_module_strict=False, load_optimizer_states=True, load_lr_scheduler_states=False,
            custom_load_fn=lambda src, dst: None)
        if loaded is None:
            raise ValueError('DeepSpeed optimizer checkpoint not found')
    else:
        optimizer.load_state_dict(torch.load(path / f'optimizer-rank{rank}.pt', map_location='cpu', weights_only=False))
    # DS optimizer restoration also restores master/partitioned trainable weights.
    # Compare its actual restored bytes; do not overwrite a mismatch with portable weights.
    if zero3:
        actual = trainable_state(unwrapped, rank, True)
        if rank == 0:
            assert_portable_equal(saved, actual)
    else:
        restore_trainable(unwrapped, saved)
    scheduler.load_state_dict(torch.load(path / 'scheduler.pt', map_location='cpu', weights_only=True))
    saved_rng = torch.load(path / f'rng-rank{rank}.pt', map_location='cpu', weights_only=False)
    restore_rng(saved_rng)
    assert_rng_equal(saved_rng, capture_rng())
    evidence = optimizer_evidence(optimizer)
    if evidence['optimizer_state_sha256'] != writer['optimizer']['optimizer_state_sha256']:
        raise ValueError('Strict restored optimizer state identity differs')
    if metadata['cursor']['update'] and not all(evidence[k] for k in ('optimizer_state_initialized', 'moments_fp32', 'moments_finite')):
        raise ValueError('Restored Adam moments absent/nonfinite/not FP32')
    if zero3 or writer['nonce'] != PROCESS_IDENTITY['nonce']:
        atomic_json(path.parent / f'resume-from-{metadata["cursor"]["update"]:07d}-rank{rank}.json',
            dict(source_checkpoint=str(path.resolve()), writer=writer, reader=PROCESS_IDENTITY,
                new_process=writer['nonce'] != PROCESS_IDENTITY['nonce'], strict_trainable_verified=True,
                optimizer=evidence, scheduler_restored=True, rng_restored=True, restored_cursor=metadata['cursor']))
    barrier()
    return metadata['cursor']


def training_outputs(args, state, data, checkpoint, fingerprint, starting_update, world):
    checkpoint = Path(checkpoint).resolve()
    marker = json.loads((checkpoint / 'trainer_state.json').read_text(encoding='utf-8'))
    if marker['fingerprint'] != fingerprint or marker['cursor'] != state or not marker['complete']:
        raise ValueError('Training result checkpoint/cursor identity differs')
    gradient_refs, resume_refs = [], []
    nonzero = dict(language_lora=0, audio_bridge=0)
    for rank in range(world):
        path = args.out / f'backward-evidence-u{starting_update:07d}-rank{rank}.json'
        evidence = json.loads(path.read_text(encoding='utf-8'))
        if (evidence['starting_update'] != starting_update or not evidence['actual_backward_observed'] or
            not sum(evidence.get('actual_checkpoint_calls', {}).values())):
            raise ValueError('Training result lacks real backward evidence')
        for name, norm in evidence['gradient_norms'].items():
            if not math.isfinite(norm):
                raise ValueError('Training result nonfinite gradient evidence')
            if 'jev_head' in name.split('.'):
                continue
            audio = 'lora_' not in name or any(tag in name.lower() for tag in ('audio', 'acoustic', 'sound'))
            nonzero['audio_bridge' if audio else 'language_lora'] += int(norm > 0)
        gradient_refs.append(file_ref(path))
        if args.resume:
            path = args.out / f'resume-from-{starting_update:07d}-rank{rank}.json'
            evidence = json.loads(path.read_text(encoding='utf-8'))
            if (not evidence['new_process'] or not evidence['strict_trainable_verified'] or
                not evidence['rng_restored'] or not evidence['scheduler_restored'] or
                not all(evidence['optimizer'][key] for key in ('optimizer_state_initialized', 'moments_fp32', 'moments_finite'))):
                raise ValueError('Training result lacks strict new-process restoration evidence')
            resume_refs.append(file_ref(path))
    if not all(nonzero.values()):
        raise ValueError('Training result lacks nonzero language LoRA/audio bridge gradients')
    mode = 'full' if not args.max_updates else 'resume_smoke' if args.resume else 'smoke'
    lifecycle = ZERO3_PARAMETER_LIFECYCLE if args.backend == 'deepspeed' else 'not-applicable'
    smoke_validation_ref = None
    if args.max_updates:
        path = args.out / f'smoke-validation-u{state["update"]:07d}.json'
        validation = json.loads(path.read_text(encoding='utf-8'))
        if (validation.get('schema') != 'music-planner-smoke-validation-v1' or
            validation.get('family') != args.family or validation.get('fingerprint') != fingerprint or
            validation.get('cursor') != state or validation.get('source_count') != data['val']['sources'] or
            validation.get('source_val_sha256') != data['val']['sha256'] or
            validation.get('phase') != 'diagnostic-only-not-checkpoint-selection' or
            validation.get('checkpoint_selection_performed') is not False or validation.get('test_used') is not False or
            validation.get('comparison_scope') != 'within-family-tokenizer-only' or
            validation.get('zero3_parameter_lifecycle') != lifecycle or
            validation.get('smoke_validation_before_save') is not True or validation.get('nll_finite') is not True or
            not math.isfinite(validation.get('nll', float('nan'))) or validation.get('assistant_tokens', 0) <= 0):
            raise ValueError('Training result lacks matching full-val-before-save smoke evidence')
        smoke_validation_ref = file_ref(path)
    continued = bool(args.resume) and state['update'] > starting_update
    result = dict(schema='music-planner-outputs-v1', family=args.family, mode=mode, world_size=world,
        optimizer_updates=state['update'], invocation_optimizer_updates=state['update'] - starting_update,
        starting_update=starting_update, cursor=state, global_batch=args.batch, epoch_budget=args.epochs,
        source_exposure_budget=data['train']['sources'] * args.epochs,
        train_count=data['train']['sources'], val_count=data['val']['sources'],
        epochs_completed=state['epoch'], source_exposures=state['source_exposures'],
        source_sha256=dict(train=data['train']['sha256'], val=data['val']['sha256']), fingerprint=fingerprint,
        run_config=file_ref(args.out / 'run_config.json'), zero3_parameter_lifecycle=lifecycle,
        smoke_validation_before_save=True, smoke_validation=smoke_validation_ref,
        gradient_groups={key: dict(finite=True, nonzero=count > 0, observed_parameters=count) for key, count in nonzero.items()},
        gradient_evidence=gradient_refs, checkpoint_complete=True, checkpoint=file_ref(checkpoint / 'trainer_state.json'),
        trainable=file_ref(checkpoint / 'trainable.pt'), checkpoint_files=marker['files'],
        resume_verified=continued, verified_new_process=continued, resume_evidence=resume_refs,
        resume_source_checkpoint=str(args.resume.resolve()) if args.resume else None,
        resumed_next_update_performed=continued, next_update_exact_match_verified=False,
        initialization='official_base_then_matching_checkpoint' if args.resume else 'official_base',
        full_budget_completed=state['epoch'] == args.epochs, test_selection_performed=False, quality_accepted=False,
        process=PROCESS_IDENTITY)
    if args.genre_classes is not None:
        result['jev'] = dict(genre_weight=args.genre_weight,
            known_label_source_exposures=state.get('genre_labeled_exposures', 0),
            head_backward_exercised=state.get('jev_head_exercised', False),
            status='head_backward_exercised' if state.get('jev_head_exercised') else 'head_not_exercised',
            music_or_choreography_quality_accepted=False)
    return result


def local_model_identity(path, weights_manifest=None):
    path = Path(path).resolve()
    if not path.is_dir() or not (path / 'config.json').is_file():
        raise ValueError('Model must be an existing local weights directory (downloads disabled)')
    weights = sorted(set(path.glob('*.safetensors')) | set(path.glob('*.bin')))
    if not weights:
        raise ValueError('No local model weights found')
    result = dict(path=str(path), config_sha256=file_sha(path / 'config.json'),
        weights=[dict(name=p.name, size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in weights],
        identity_method='local-file-stat-plus-config-SHA256; weight-content-SHA-not-claimed')
    if weights_manifest:
        manifest_path = Path(weights_manifest).resolve()
        result['weights_manifest'] = dict(path=str(manifest_path), sha256=file_sha(manifest_path))
        result['identity_method'] += '+resource-weight-manifest'
    for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json', 'tokenizer.json',
                 'tokenizer_config.json', 'preprocessor_config.json', 'processor_config.json', 'chat_template.jinja'):
        if (path / name).is_file():
            result[name] = file_sha(path / name)
    return result


def resolve_adapter():
    try:
        module = importlib.import_module('scripts_omnidance.exp.ms.music_planner_adapters')
    except ModuleNotFoundError as error:
        if error.name == 'scripts_omnidance':
            module = importlib.import_module('music_planner_adapters')
        else:
            raise RuntimeError(f'BLOCKED: music planner adapter dependency unavailable: {error}') from error
    if not hasattr(module, 'load_planner_adapter'):
        raise RuntimeError('BLOCKED: adapter module must export load_planner_adapter')
    return module.load_planner_adapter


def deepspeed_config(args, world):
    config = json.loads(Path(args.deepspeed).read_text(encoding='utf-8')) if args.deepspeed else {}
    zero = config.setdefault('zero_optimization', {})
    if zero.get('stage', 3) != 3:
        raise ValueError('Music planner distributed backend requires ZeRO3')
    zero.update(stage=3, overlap_comm=True, contiguous_gradients=True)
    for key in ('fp16', 'bf16'):
        if config.get(key, {}).get('enabled') is True:
            raise ValueError('DS precision conversion would downcast FP32 LoRA; use native BF16 torch_autocast')
        config[key] = {'enabled': False}
    if 'optimizer' in config or 'scheduler' in config:
        raise ValueError('Optimizer/scheduler are explicit FP32 AdamW, not configured by DeepSpeed')
    config.update(torch_autocast=dict(enabled=True, dtype='bfloat16'), communication_data_type='fp32',
                  train_micro_batch_size_per_gpu=args.micro, gradient_accumulation_steps=args.accum,
                  train_batch_size=args.micro * args.accum * world, gradient_clipping=args.grad_clip)
    return config


def prepare_batch(adapter, rows, cache, empty=False):
    for row in rows:
        cache.verify(row)
    batch = adapter.collate(rows)
    if not isinstance(batch, dict) or 'labels' not in batch:
        raise ValueError('Adapter must return tensor batch with assistant-only labels')
    if not empty and not bool(batch['labels'][:, 1:].ne(-100).any()):
        raise ValueError('Adapter produced no causal assistant/EOS labels')
    if empty:
        batch['labels'] = torch.full_like(batch['labels'], -100)
    classes = getattr(adapter, 'jev_classes', None)
    if classes is not None:
        targets, mask = planner_module('music_planner_jev').batch_jev_targets(rows, classes)
        batch['jev_targets'] = torch.zeros_like(targets) if empty else targets
        batch['jev_label_mask'] = torch.zeros_like(mask) if empty else mask
    return batch


def move_batch(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items() if key not in ('labels', 'jev_targets', 'jev_label_mask')}


def genre_batch_loss(output, batch, device):
    if 'jev_targets' not in batch:
        return None, torch.zeros((), dtype=torch.long, device=device)
    jev = planner_module('music_planner_jev')
    logits = jev.get_jev_logits(output)
    if logits is None:
        raise ValueError('Installed JEV did not produce native-prefix genre logits')
    return jev.jev_loss(logits, batch['jev_targets'].to(device), batch['jev_label_mask'].to(device))


def evaluate(model, adapter, rows, cache, accelerator, micro, native_autocast=False, zero3=False,
             loss_chunk_size=0):
    model.eval()
    total = torch.zeros(4, dtype=torch.float64, device=accelerator.device)
    world, rank = accelerator.num_processes, accelerator.process_index
    try:
        for start in range(0, len(rows), micro * world):
            selected = rows[start + rank * micro:start + (rank + 1) * micro]
            batch = prepare_batch(adapter, selected or [rows[0]], cache, empty=not selected)
            autocast = nullcontext() if native_autocast else torch.autocast(accelerator.device.type, dtype=torch.bfloat16)
            with torch.no_grad(), autocast:
                output = model(**move_batch(batch, accelerator.device))
                loss, count = shifted_nll(output.logits, batch['labels'].to(accelerator.device), loss_chunk_size)
                genre_sum, genre_count = genre_batch_loss(output, batch, accelerator.device)
            total += torch.stack((loss.detach().double(), count.double(),
                                  genre_sum.detach().double() if genre_sum is not None else loss.detach().double() * 0,
                                  genre_count.double()))
        total = accelerator.reduce(total, reduction='sum')
    finally:
        finish_zero3_forward(model, zero3)
        model.train()
    if total[1].item() <= 0 or not torch.isfinite(total[0]):
        raise ValueError('Invalid within-family validation NLL')
    result = dict(nll=(total[0] / total[1]).item(), assistant_tokens=int(total[1].item()),
                  source_count=len(rows), comparison_scope='within-family-tokenizer-only', test_used=False)
    if getattr(adapter, 'jev_classes', None) is not None:
        result.update(genre_nll=(total[2] / total[3]).item() if total[3].item() else None,
                      genre_labeled_sources=int(total[3].item()))
    return result


def validate_smoke_before_save(args, model, adapter, rows, cache, accelerator, state, data, fingerprint, zero3=False):
    validation = evaluate(model, adapter, rows, cache, accelerator, args.micro,
                          native_autocast=zero3, zero3=zero3, loss_chunk_size=args.loss_chunk_size)
    path = args.out / f'smoke-validation-u{state["update"]:07d}.json'
    if accelerator.is_main_process:
        atomic_json(path, dict(schema='music-planner-smoke-validation-v1', family=args.family,
            phase='diagnostic-only-not-checkpoint-selection', checkpoint_selection_performed=False,
            cursor=dict(state), source_val_sha256=data['val']['sha256'], fingerprint=fingerprint,
            zero3_parameter_lifecycle=ZERO3_PARAMETER_LIFECYCLE if zero3 else 'not-applicable',
            smoke_validation_before_save=True, nll_finite=math.isfinite(validation['nll']), **validation))
    return path


def train(args):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    schema = planner_module('music_planner_schema')
    # Keep the historical loader directly callable for existing fair fixtures.
    train_rows, val_rows, data = (schema.read_training_data(args)
        if args.caption_contract is not None or args.genre_classes is not None or
           args.train_genre_labels is not None or args.val_genre_labels is not None
        else check_data(args.train, args.val))
    if args.check_data_only:
        print(json.dumps(dict(data=data, actual_training_started=False), ensure_ascii=False), flush=True)
        return
    if args.backend == 'deepspeed' and args.verify_reload:
        raise ValueError('ZeRO3 --verify-reload same-engine is unsupported; run fresh --resume process instead')
    identity = local_model_identity(args.model, args.weights_manifest)
    loader = resolve_adapter()
    try:
        from accelerate import Accelerator, DeepSpeedPlugin
        from accelerate.utils import GradientAccumulationPlugin, set_seed
    except ImportError as error:
        raise RuntimeError(f'BLOCKED: existing planner environment lacks accelerate/deepspeed: {error}') from error
    world = int(os.environ.get('WORLD_SIZE', '1'))
    cluster = cluster_environment(world)
    if args.micro != 1:
        raise ValueError('Native audio feature packing currently requires --micro 1')
    global_batch = args.micro * args.accum * world
    if args.batch != global_batch:
        raise ValueError(f'Global batch must equal micro*accum*world={global_batch}, not {args.batch}')
    ds_config = deepspeed_config(args, world) if args.backend == 'deepspeed' else None
    plugin = DeepSpeedPlugin(hf_ds_config=ds_config, zero3_init_flag=True) if ds_config else None
    accelerator = Accelerator(mixed_precision='no', deepspeed_plugin=plugin,
        gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=args.accum, sync_with_dataloader=False))
    rank, world = accelerator.process_index, accelerator.num_processes
    if cluster is not None and world != cluster['world_size']:
        raise ValueError('Initialized distributed world differs from the launch contract')
    identity = verify_cluster_model_identity(args.model, identity, accelerator, cluster)
    set_seed(args.seed)
    if accelerator.is_main_process:
        if args.out.exists() and not args.resume:
            raise ValueError('Output exists; use --resume rather than overwrite')
        args.out.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    with plugin.zero3_init_context_manager(enable=True) if plugin else nullcontext():
        adapter = loader(family=args.family, model_path=args.model, lora_rank=args.rank,
                         lora_alpha=args.alpha, max_length=args.max_length, audio_last_n=args.audio_last_n)
        adapter = schema.configure_adapter(adapter, args)
        if args.genre_classes is not None:
            classes = data['jev']['classes']
            adapter.jev_metadata = planner_module('music_planner_jev').install_jev(
                adapter, classes, head_dim=args.genre_head_dim)
            adapter.jev_classes = classes
    model = adapter.model
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
    if not any(getattr(module, 'gradient_checkpointing', False) for module in model.modules()):
        raise RuntimeError('BLOCKED: architecture did not actually enable gradient checkpointing')
    base_dtype = prepare_training_storage(model, loader, bool(plugin))
    moe_leaves = configure_zero3_leaf_modules(model, args.family, bool(plugin))
    groups, group_names = optimizer_groups(model, args.lora_lr, args.projector_lr,
                                          base_dtype=base_dtype, genre_lr=args.genre_lr)
    optimizer = torch.optim.AdamW(groups, weight_decay=0.)
    total_updates = math.ceil(len(train_rows) / args.batch) * args.epochs
    warmup_updates = (math.ceil(total_updates * args.warmup_ratio)
                      if args.warmup_ratio is not None else args.warmup_updates)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: lr_multiplier(step, total_updates, warmup_updates, args.lr_schedule))
    source_names = ['music_planner_train', 'music_planner_token_loss', 'music_planner_schema']
    if cluster is not None:
        source_names.append('music_planner_cluster')
    if args.genre_classes is not None:
        source_names.append('music_planner_jev')
    source_hashes = {name: file_sha(__file__ if name == 'music_planner_train' else planner_module(name).__file__)
                     for name in source_names}
    config = dict(family=args.family, model=identity, data=data, epochs=args.epochs, total_updates=total_updates,
        source_exposure_budget=len(train_rows) * args.epochs, global_batch=args.batch, micro=args.micro, accum=args.accum,
        world_size=world, seed=args.seed, lora_rank=args.rank, lora_alpha=args.alpha, lora_lr=args.lora_lr,
        projector_lr=args.projector_lr, grad_clip=args.grad_clip, warmup_updates=args.warmup_updates,
        max_length=args.max_length, backend=args.backend, base_dtype='float32' if plugin else 'bf16',
        load_dtype='bf16', compute_dtype='bf16', trainable_dtype='float32',
        adapter_source_sha256=file_sha(importlib.import_module(loader.__module__).__file__),
        optimizer='AdamW-FP32-full-states', gradient_checkpointing=True, checkpointing_use_reentrant=True, parameter_groups=group_names,
        trainable_signature=parameter_signature(model), trainer_sha256=file_sha(__file__),
        zero3_parameter_lifecycle=ZERO3_PARAMETER_LIFECYCLE if plugin else 'not-applicable',
        smoke_validation_before_save=True, source_sha256=source_hashes,
        resolved_deepspeed_config=ds_config, zero3_moe_leaves=moe_leaves,
        audio_last_n=args.audio_last_n, loss_chunk_size=args.loss_chunk_size,
        loss_policy='global-assistant-token-mean-plus-independent-global-labeled-source-genre-mean-v1',
        genre_weight=args.genre_weight, genre_lr=args.genre_lr, genre_head_dim=args.genre_head_dim,
        jev=getattr(adapter, 'jev_metadata', None), caption_contract=data.get('contract'),
        lr_schedule=args.lr_schedule, warmup_ratio=args.warmup_ratio, effective_warmup_updates=warmup_updates,
        validation_every_updates=args.eval_every, save_every_updates=args.save_every, cluster=cluster,
        software=software_identity())
    fingerprint = object_sha(config)
    assert_shared_fingerprint(fingerprint)
    if accelerator.is_main_process:
        config_path = args.out / 'run_config.json'
        if args.resume:
            existing = json.loads(config_path.read_text(encoding='utf-8'))
            canonical = {key: value for key, value in existing.items() if key not in ('fingerprint', 'trainable_parameters')}
            if existing.get('fingerprint') != fingerprint or object_sha(canonical) != fingerprint:
                raise ValueError('Resume run config differs; original run config was not overwritten')
        else:
            atomic_json(config_path, dict(config, fingerprint=fingerprint,
                trainable_parameters=sum(math.prod(v['shape']) for v in config['trainable_signature'].values())))
    model, optimizer = accelerator.prepare(model, optimizer)
    # Scheduler is deliberately not Accelerator-prepared: one scheduler tick per actual optimizer update.
    state = dict(epoch=0, position=0, update=0, source_exposures=0, best_val_nll=None)
    if args.genre_classes is not None:
        state.update(genre_labeled_exposures=0, jev_head_exercised=False)
    checkpoint_args = dict(rank=rank, world=world, zero3=bool(plugin), barrier=accelerator.wait_for_everyone)
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, fingerprint, **checkpoint_args)
    validate_cursor(state, len(train_rows), args.epochs, args.batch)
    starting_update = state['update']
    gradient_observe_until = starting_update + 2
    cache = AudioIdentityCache(args.out / f'audio_identity_cache-rank{rank}.json')
    last_checkpoint = args.resume.resolve() if args.resume else None
    def save():
        nonlocal last_checkpoint
        path = args.out.resolve() / f'checkpoint-{state["update"]:07d}'
        cache.flush()
        if path == last_checkpoint:
            return path
        save_checkpoint(path, model, optimizer, scheduler, state, fingerprint, adapter=adapter, **checkpoint_args)
        last_checkpoint = path
        if accelerator.is_main_process:
            atomic_json(args.out / 'latest.json', dict(checkpoint=str(path), update=state['update'], cursor=state))
            if args.eval_every and state['update'] % args.eval_every == 0:
                atomic_json(path / 'val_generation_request.json', dict(checkpoint=str(path), family=args.family,
                    val_manifest=data['val'], text_panel_size=32, video_followup_cases=3,
                    split='val', test_used=False))
        return path
    model.train()
    checkpoint_calls = {}
    def observe_checkpoint(name, original):
        def wrapped(*positional, **keywords):
            if state['update'] < gradient_observe_until:
                checkpoint_calls[name] = checkpoint_calls.get(name, 0) + 1
            return original(*positional, **keywords)
        return wrapped
    unwrapped = model.module if hasattr(model, 'module') else model
    for name, module in unwrapped.named_modules():
        original = getattr(module, '_gradient_checkpointing_func', None)
        if getattr(module, 'gradient_checkpointing', False) and callable(original):
            module._gradient_checkpointing_func = observe_checkpoint(name, original)
    observed_gradients = {}
    hook_handles = []
    def observe(name):
        def hook(gradient):
            if state['update'] < gradient_observe_until or (
                    'jev_head' in name.split('.') and not state.get('jev_head_exercised', False)):
                if not bool(torch.isfinite(gradient).all()):
                    raise FloatingPointError(f'Nonfinite real backward gradient: {name}')
                observed_gradients[name] = max(observed_gradients.get(name, 0.), float(gradient.float().norm()))
        return hook
    unwrapped = model.module if hasattr(model, 'module') else model
    for name, parameter in unwrapped.named_parameters():
        if parameter.requires_grad:
            hook_handles.append((parameter.register_hook(observe(name)), 'jev_head' in name.split('.')))
    optimizer.zero_grad(set_to_none=True)
    while state['epoch'] < args.epochs and (not args.max_updates or state['update'] < args.max_updates):
        order = epoch_order(len(train_rows), args.seed, state['epoch'])
        slices, consumed = update_slices(order, state['position'], args.micro, args.accum, rank, world)
        batches = [prepare_batch(adapter, [train_rows[i] for i in indices] or [train_rows[order[state['position']]]],
                                 cache, empty=not indices) for indices in slices]
        local_support = torch.tensor([
            sum(int(batch['labels'][:, 1:].ne(-100).sum()) for batch in batches),
            sum(int(batch['jev_label_mask'].sum()) for batch in batches if 'jev_label_mask' in batch)],
            dtype=torch.float64, device=accelerator.device)
        global_tokens, global_genres = accelerator.reduce(local_support, reduction='sum').unbind()
        if global_tokens.item() <= 0:
            raise ValueError('No valid assistant tokens in optimizer update')
        update_loss = torch.zeros((), device=accelerator.device, dtype=torch.float64)
        update_genre_loss = torch.zeros_like(update_loss)
        for batch in batches:
            with accelerator.accumulate(model):
                autocast = nullcontext() if plugin else torch.autocast(accelerator.device.type, dtype=torch.bfloat16)
                with autocast:
                    output = model(**move_batch(batch, accelerator.device))
                    nll, _ = shifted_nll(output.logits, batch['labels'].to(accelerator.device), args.loss_chunk_size)
                    genre_sum, _ = genre_batch_loss(output, batch, accelerator.device)
                    if not bool(torch.isfinite(nll)):
                        raise FloatingPointError('Nonfinite assistant-token training NLL')
                    loss = planner_module('music_planner_token_loss').joint_update_objective(
                        nll, genre_sum, global_tokens, global_genres, world=world,
                        accum=args.accum, genre_weight=args.genre_weight)
                accelerator.backward(loss)
                # DeepSpeed clips within its managed optimizer boundary. Accelerate's
                # DeepSpeed backward already performs engine.step; do not clip twice.
                if accelerator.sync_gradients and not plugin:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            update_loss += nll.detach().double()
            if genre_sum is not None:
                update_genre_loss += genre_sum.detach().double()
        scheduler.step()
        if state['update'] < gradient_observe_until:
            categories = sorted(group_names)
            nonzero = torch.tensor([sum(observed_gradients.get(name, 0.) > 0 for name in group_names[kind])
                                    for kind in categories], device=accelerator.device)
            nonzero = accelerator.reduce(nonzero, reduction='sum')
            required = [kind for kind in categories if group_names[kind] and
                        (kind != 'jev' or (args.genre_weight > 0 and global_genres.item() > 0))]
            if any(count == 0 for kind, count in zip(categories, nonzero.tolist()) if kind in required):
                raise ValueError('Missing nonzero real backward gradient in required trainable group')
            if not sum(checkpoint_calls.values()):
                raise ValueError('Architecture enabled checkpoint flags but no actual checkpoint call occurred')
            atomic_json(args.out / f'backward-evidence-u{starting_update:07d}-rank{rank}.json',
                dict(starting_update=starting_update, update=state['update'] + 1,
                gradient_norms=observed_gradients, nonzero_by_group=dict(zip(categories, nonzero.tolist())),
                gradient_checkpointing_enabled=True, actual_checkpoint_calls=checkpoint_calls, actual_backward_observed=True))
        if args.genre_classes is not None:
            state['genre_labeled_exposures'] += int(global_genres)
            if args.genre_weight > 0 and global_genres.item() > 0 and not state['jev_head_exercised']:
                local_nonzero = sum(observed_gradients.get(name, 0.) > 0 for name in group_names['jev'])
                global_nonzero = accelerator.reduce(torch.tensor(local_nonzero, device=accelerator.device), reduction='sum')
                if global_nonzero.item() <= 0:
                    raise ValueError('Labeled JEV update produced no real nonzero head gradient')
                atomic_json(args.out / f'jev-backward-u{state["update"] + 1:07d}-rank{rank}.json',
                    dict(update=state['update'] + 1, labeled_sources_in_update=int(global_genres),
                         global_nonzero_head_parameters=int(global_nonzero),
                         local_head_gradient_norms={name: observed_gradients.get(name, 0.) for name in group_names['jev']},
                         head_backward_exercised=True, genre_weight=args.genre_weight))
                state['jev_head_exercised'] = True
                state['jev_first_exercised_update'] = state['update'] + 1
        state['update'] += 1
        if state['update'] >= gradient_observe_until:
            remaining = []
            for handle, is_head in hook_handles:
                if is_head and not state.get('jev_head_exercised', False):
                    remaining.append((handle, is_head))
                else:
                    handle.remove()
            hook_handles = remaining
        state['source_exposures'] += consumed
        state['position'] += consumed
        epoch_finished = state['position'] == len(train_rows)
        if epoch_finished:
            state['epoch'] += 1
            state['position'] = 0
        should_validate = not args.max_updates and (epoch_finished or
                            (args.eval_every and state['update'] % args.eval_every == 0))
        if should_validate:
            validation = evaluate(model, adapter, val_rows, cache, accelerator, args.micro,
                                  native_autocast=bool(plugin), zero3=bool(plugin), loss_chunk_size=args.loss_chunk_size)
            improved = state['best_val_nll'] is None or validation['nll'] < state['best_val_nll']
            if improved:
                state['best_val_nll'] = validation['nll']
            checkpoint = save()
            if accelerator.is_main_process:
                atomic_json(checkpoint / 'validation.json', validation)
                if improved:
                    atomic_json(args.out / 'best.json', dict(checkpoint=str(checkpoint), **validation))
        elif (args.save_every and state['update'] % args.save_every == 0 and
              not (args.max_updates and state['update'] >= args.max_updates)):
            save()
        update_loss, update_genre_loss = accelerator.reduce(
            torch.stack((update_loss, update_genre_loss)), reduction='sum').unbind()
        if accelerator.is_main_process:
            print(json.dumps(dict(update=state['update'], epoch=state['epoch'], position=state['position'],
                source_exposures=state['source_exposures'], train_token_nll=(update_loss / global_tokens).item(),
                train_genre_nll=(update_genre_loss / global_genres).item() if global_genres.item() else None,
                global_assistant_tokens=int(global_tokens), global_genre_labeled_sources=int(global_genres))), flush=True)
    if args.max_updates:
        validate_smoke_before_save(args, model, adapter, val_rows, cache, accelerator,
                                   state, data, fingerprint, zero3=bool(plugin))
    checkpoint = save()
    if args.max_updates and args.verify_reload:
        restored = load_checkpoint(checkpoint, model, optimizer, scheduler, fingerprint, **checkpoint_args)
        if restored != state:
            raise ValueError('Smoke checkpoint cursor roundtrip failed')
        unwrapped = model.module if hasattr(model, 'module') else model
        actual = trainable_state(unwrapped, rank, bool(plugin))
        if accelerator.is_main_process:
            expected = torch.load(checkpoint / 'trainable.pt', map_location='cpu', weights_only=True)
            for name in expected:
                if not torch.equal(expected[name], actual[name]):
                    raise ValueError(f'Smoke trainable reload differs: {name}')
            saved_rng = torch.load(checkpoint / 'rng-rank0.pt', map_location='cpu', weights_only=False)
            actual_rng = capture_rng()
            if saved_rng['python'] != actual_rng['python'] or not torch.equal(saved_rng['torch'], actual_rng['torch']):
                raise ValueError('Smoke RNG reload differs')
            if 'cuda' in saved_rng and not all(torch.equal(a, b) for a, b in zip(saved_rng['cuda'], actual_rng['cuda'])):
                raise ValueError('Smoke CUDA RNG reload differs')
            atomic_json(args.out / 'smoke_reload.json', dict(checkpoint=str(checkpoint), updates=state['update'],
                save_reload_passed=True, trainable_reload_exact=True, optimizer_scheduler_rng_restored=True,
                next_update_exact_proof='CPU fixture only; real family continuation still needs resumed update',
                full_training_completed=False))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        outputs = training_outputs(args, state, data, checkpoint, fingerprint, starting_update, world)
        atomic_json(args.out / f'outputs-u{state["update"]:07d}.json', outputs)
        atomic_json(args.out / 'outputs.json', outputs)
        atomic_json(args.out / 'training_status.json', dict(cursor=state, checkpoint=str(checkpoint),
            full_budget_completed=state['epoch'] == args.epochs, quality_accepted=False,
            smoke_only=bool(args.max_updates), test_used=False))
    accelerator.wait_for_everyone()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--family', choices=FAMILIES, required=True)
    result.add_argument('--model', type=Path, required=True)
    result.add_argument('--train', type=Path, required=True)
    result.add_argument('--val', type=Path, required=True)
    result.add_argument('--out', type=Path, required=True)
    result.add_argument('--epochs', type=int, default=2)
    result.add_argument('--max-updates', type=int, default=0, help='Absolute optimizer-update ceiling for smoke; 0 means full budget')
    result.add_argument('--batch', type=int, default=32, help='Global source batch, not per GPU')
    result.add_argument('--micro', type=int, default=1)
    result.add_argument('--accum', type=int, default=4)
    result.add_argument('--lora-lr', type=float, default=1e-5)
    result.add_argument('--projector-lr', type=float, default=5e-5)
    result.add_argument('--rank', type=int, default=32)
    result.add_argument('--alpha', type=int, default=64)
    result.add_argument('--grad-clip', type=float, default=1.)
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--warmup-updates', type=int, default=10)
    result.add_argument('--max-length', type=int, default=8192)
    result.add_argument('--audio-last-n', type=int, default=4, help='Native audio-tower trailing LoRA layers')
    result.add_argument('--loss-chunk-size', type=int, default=128, help='Assistant CE rows per recomputed vocabulary chunk; 0 disables chunk recompute')
    result.add_argument('--caption-contract', type=Path, help='Explicit versioned full-audio/text schema; absent keeps historical fixed2s data')
    result.add_argument('--genre-classes', type=Path, help='Explicit JSON class list; never inferred from captions')
    result.add_argument('--train-genre-labels', type=Path)
    result.add_argument('--val-genre-labels', type=Path)
    result.add_argument('--genre-weight', type=float, default=0., help='Independent global labeled-source genre CE coefficient')
    result.add_argument('--genre-lr', type=float, default=1e-4)
    result.add_argument('--genre-head-dim', type=int, default=256)
    result.add_argument('--lr-schedule', choices=('constant', 'cosine'), default='constant')
    result.add_argument('--warmup-ratio', type=float, help='Override warmup-updates with ceil(total updates * ratio)')
    result.add_argument('--backend', choices=('deepspeed', 'ddp'), default='deepspeed')
    result.add_argument('--deepspeed', type=Path, help='Optional ZeRO3 config; default built in')
    result.add_argument('--resume', type=Path)
    result.add_argument('--weights-manifest', type=Path)
    result.add_argument('--save-every', type=int, default=100)
    result.add_argument('--eval-every', type=int, default=100)
    result.add_argument('--verify-reload', action=argparse.BooleanOptionalAction, default=False,
                        help='DDP only same-process roundtrip; ZeRO3 uses a separate --resume invocation')
    result.add_argument('--check-data-only', action='store_true')
    return result


def main():
    args = parser().parse_args()
    if min(args.epochs, args.batch, args.micro, args.accum, args.rank, args.alpha, args.max_length) < 1:
        raise ValueError('Positive epochs/batch/micro/accum/rank/alpha/context required')
    if args.max_updates < 0 or args.save_every < 0 or args.eval_every < 0:
        raise ValueError('Update limits/intervals must not be negative')
    if min(args.audio_last_n, args.genre_head_dim) < 1 or args.loss_chunk_size < 0:
        raise ValueError('Positive audio scope/head size and nonnegative loss chunk size required')
    if (any(not math.isfinite(value) or value <= 0 for value in
            (args.lora_lr, args.projector_lr, args.genre_lr, args.grad_clip)) or
            not math.isfinite(args.genre_weight) or args.genre_weight < 0 or args.warmup_updates < 0):
        raise ValueError('Finite positive learning rates/clip and nonnegative genre weight/warmup required')
    if args.warmup_ratio is not None and (not math.isfinite(args.warmup_ratio) or not 0 <= args.warmup_ratio < 1):
        raise ValueError('warmup-ratio must be finite in [0,1)')
    if args.genre_weight and args.genre_classes is None:
        raise ValueError('Positive genre loss requires the user-supplied --genre-classes')
    train(args)


if __name__ == '__main__':
    main()
