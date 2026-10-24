import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch

ENTRY = Path(__file__).resolve().parents[1] / 'scripts_omnidance/exp/ms/music_planner_train.py'
spec = importlib.util.spec_from_file_location('music_planner_train_tested', ENTRY)
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def row(tmp_path, sid, split):
    audio = tmp_path / f'{sid}.wav'
    audio.write_bytes(sid.encode())
    return dict(id=sid, source_group_id=f'group-{sid}', audio_path=str(audio),
        audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(), split=split,
        audio_window_seconds=[0., 5.0625], target_text='[0.000-2.000s] sway\n[2.000-5.062s] step',
        target_windows=[dict(start=0., end=2., caption='sway'), dict(start=2., end=5.0625, caption='step')],
        caption_version='fixed2s-weak-v1', semantic_approval=False, human_review_completed=False)


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def test_data_boundary_counts_sources_not_windows(tmp_path):
    train_path, val_path = tmp_path / 'train.jsonl', tmp_path / 'val.jsonl'
    write_rows(train_path, [row(tmp_path, 'a', 'train'), row(tmp_path, 'b', 'train')])
    write_rows(val_path, [row(tmp_path, 'c', 'val')])
    training, val, identity = train.check_data(train_path, val_path)
    assert len(training) == 2 and len(val) == 1
    assert identity['train']['sources'] == 2
    assert identity['train']['sha256'] == train.file_sha(train_path)
    assert identity['test_used'] is False
    assert identity['song_near_duplicate_status'] == 'unknown'


@pytest.mark.parametrize('mutation', ['test', 'same_source', 'same_audio', 'crop', 'approval'])
def test_bad_boundary_rejected(tmp_path, mutation):
    a, b = row(tmp_path, 'a', 'train'), row(tmp_path, 'b', 'val')
    if mutation == 'test':
        b['split'] = 'test'
    elif mutation == 'same_source':
        b['source_group_id'] = a['source_group_id']
    elif mutation == 'same_audio':
        b['audio_sha256'] = a['audio_sha256']
    elif mutation == 'crop':
        a['audio_window_seconds'] = [0., 2.]
    else:
        a['semantic_approval'] = True
    write_rows(tmp_path / 'train.jsonl', [a])
    write_rows(tmp_path / 'val.jsonl', [b])
    with pytest.raises(ValueError):
        train.check_data(tmp_path / 'train.jsonl', tmp_path / 'val.jsonl')


def test_isolation_group_can_have_multiple_complete_source_ids(tmp_path):
    a, b = row(tmp_path, 'a', 'train'), row(tmp_path, 'b', 'train')
    b['source_group_id'] = a['source_group_id']
    write_rows(tmp_path / 'train.jsonl', [a, b])
    rows, identity = train.read_split(tmp_path / 'train.jsonl', 'train')
    assert len(rows) == identity['sources'] == 2
    assert identity['isolation_groups'] == 1


def test_same_source_two_windows_cannot_be_two_examples(tmp_path):
    a = row(tmp_path, 'a', 'train')
    b = dict(a)
    write_rows(tmp_path / 'train.jsonl', [a, b])
    with pytest.raises(ValueError, match='once per epoch'):
        train.read_split(tmp_path / 'train.jsonl', 'train')


def test_split_preflight_does_not_resolve_audio_files(tmp_path, monkeypatch):
    a = row(tmp_path, 'a', 'train')
    a['audio_path'] = './a.wav'
    manifest = tmp_path / 'train.jsonl'
    write_rows(manifest, [a])
    actual_resolve = Path.resolve
    def resolve_only_manifest(path, *args, **kwargs):
        if path != manifest:
            raise AssertionError('Audio must not be probed before consumption')
        return actual_resolve(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'resolve', resolve_only_manifest)
    rows, _ = train.read_split(manifest, 'train')
    assert rows[0]['audio_path'] == str(tmp_path / 'a.wav')


def test_audio_identity_cache_does_not_hash_each_epoch(tmp_path, monkeypatch):
    a = row(tmp_path, 'a', 'train')
    cache = train.AudioIdentityCache(tmp_path / 'cache.json')
    actual_sha = train.file_sha
    calls = []
    def counted(path):
        calls.append(path)
        return actual_sha(path)
    monkeypatch.setattr(train, 'file_sha', counted)
    cache.verify(a)
    cache.verify(a)
    assert len(calls) == 1
    Path(a['audio_path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='Audio bytes changed'):
        cache.verify(a)


def test_all_sources_once_with_partial_global_batch():
    order = train.epoch_order(10720, 42, 0)
    consumed, seen = 0, []
    while consumed < len(order):
        per_rank = [train.update_slices(order, consumed, 1, 4, rank, 8) for rank in range(8)]
        counts = {count for _, count in per_rank}
        assert len(counts) == 1
        for slices, _ in per_rank:
            seen.extend(i for micro in slices for i in micro)
        consumed += per_rank[0][1]
    assert len(seen) == 10720
    assert set(seen) == set(range(10720))
    assert len(set(seen)) == len(seen)
    assert consumed == 10720
    # Non-divisible source count exercises empty rank/micro slots, not duplication/drop_last.
    tail = train.epoch_order(35, 42, 0)
    seen = []
    for position in (0, 32):
        for rank in range(8):
            slices, _ = train.update_slices(tail, position, 1, 4, rank, 8)
            assert len(slices) == 4
            seen.extend(i for indices in slices for i in indices)
    assert len(seen) == len(set(seen)) == 35


def test_cursor_strict_source_exposures():
    state = dict(epoch=1, position=32, update=336, source_exposures=10752)
    train.validate_cursor(state, 10720, 2, 32)
    with pytest.raises(ValueError, match='exposures'):
        train.validate_cursor(dict(state, source_exposures=10753), 10720, 2, 32)
    with pytest.raises(ValueError, match='boundary'):
        train.validate_cursor(dict(state, position=31), 10720, 2, 32)


def test_causal_nll_is_assistant_token_weighted_and_empty_batch_has_grad():
    logits = torch.randn(2, 4, 7, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 3], [-100, 1, 2, -100]])
    loss, count = train.shifted_nll(logits, labels)
    assert count.item() == 4
    expected = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, 7),
        labels[:, 1:].reshape(-1), ignore_index=-100, reduction='sum')
    torch.testing.assert_close(loss, expected)
    empty, count = train.shifted_nll(logits, torch.full_like(labels, -100))
    assert empty.item() == 0 and count.item() == 0
    empty.backward()
    assert logits.grad is not None and not bool(logits.grad.any())


class TinyPlanner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16), requires_grad=False)
        self.lora_A = torch.nn.Parameter(torch.randn(3, 3))
        self.audio_projector = torch.nn.Parameter(torch.randn(3, 3))
        self.dropout = torch.nn.Dropout(.2)

    def forward(self, x):
        return self.dropout(x) @ self.lora_A @ self.audio_projector


def make_optimizer(model):
    groups, names = train.optimizer_groups(model, 1e-5, 5e-5)
    assert len(names['lora']) == len(names['projector']) == 1
    opt = torch.optim.AdamW(groups, weight_decay=0.)
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda step: (step + 1) / 10)


def update(model, opt, scheduler):
    x = torch.randn(2, 3) + random.random()
    target = torch.randn(2, 3)
    loss = (model(x) - target).square().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    opt.step()
    opt.zero_grad(set_to_none=True)
    scheduler.step()
    return loss.detach()


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_two_updates_save_reload_next_step_exact_with_adam_rng_and_scheduler(tmp_path):
    torch.manual_seed(42)
    random.seed(42)
    model = TinyPlanner()
    opt, scheduler = make_optimizer(model)
    update(model, opt, scheduler)
    update(model, opt, scheduler)
    state = dict(epoch=0, position=64, update=2, source_exposures=64, best_val_nll=None)
    checkpoint = train.save_checkpoint(tmp_path / 'checkpoint-0000002', model, opt, scheduler, state, 'fixture')
    portable = torch.load(checkpoint / 'trainable.pt', weights_only=True)
    assert 'base' not in portable
    assert set(portable) == {'lora_A', 'audio_projector'}
    assert all(v.dtype == torch.float32 for v in portable.values())
    expected_loss = update(model, opt, scheduler)
    expected_weights = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(opt.state_dict())
    expected_scheduler = copy.deepcopy(scheduler.state_dict())
    clone = TinyPlanner()
    clone_opt, clone_scheduler = make_optimizer(clone)
    restored = train.load_checkpoint(checkpoint, clone, clone_opt, clone_scheduler, 'fixture')
    assert restored == state
    actual_loss = update(clone, clone_opt, clone_scheduler)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    assert_nested_equal(clone.state_dict(), expected_weights)
    assert_nested_equal(clone_opt.state_dict(), expected_optimizer)
    assert_nested_equal(clone_scheduler.state_dict(), expected_scheduler)
    for value in clone_opt.state.values():
        assert value['exp_avg'].dtype == value['exp_avg_sq'].dtype == torch.float32


@pytest.fixture
def zero3_optimizer_state_fixture():
    pytest.importorskip('deepspeed')
    from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
    from accelerate.utils.deepspeed import DeepSpeedOptimizerWrapper
    flat = [torch.nn.Parameter(torch.tensor([1., 2.])),
            torch.nn.Parameter(torch.tensor([3.])),
            torch.nn.Parameter(torch.tensor([4., 5., 6.]))]
    adam = torch.optim.AdamW([dict(params=flat[:2], lr=1e-5, name='lora'),
                             dict(params=flat[2:], lr=5e-5, name='projector')])
    sum(parameter.square().sum() for parameter in flat).backward()
    adam.step()
    adam.zero_grad(set_to_none=True)
    expected = train.optimizer_evidence(adam)
    # Exercise actual DS public serialization without initializing a GPU engine.
    zero = DeepSpeedZeroOptimizer_Stage3.__new__(DeepSpeedZeroOptimizer_Stage3)
    zero.optimizer = adam
    zero.fp16_groups = [[parameter] for parameter in flat]
    zero.sub_group_to_group_id = {0: 0, 1: 0, 2: 1}
    zero.fp32_partitioned_groups_flat = flat
    zero.loss_scaler = None
    zero.dynamic_loss_scale = False
    zero.overflow = False
    zero.partition_count = 1
    zero.elastic_checkpoint = False
    zero._clear_fp32_optimizer_param_groups()
    wrapper = DeepSpeedOptimizerWrapper.__new__(DeepSpeedOptimizerWrapper)
    wrapper.optimizer = zero
    return wrapper, adam, expected


def test_zero3_public_optimizer_evidence_handles_empty_inner_groups(zero3_optimizer_state_fixture, tmp_path):
    wrapper, adam, expected = zero3_optimizer_state_fixture
    with pytest.raises(KeyError):
        adam.state_dict()
    assert expected['optimizer_state_initialized'] is True
    assert expected['moment_tensors'] == 6
    assert expected['moments_fp32'] is expected['moments_finite'] is True
    assert train.optimizer_evidence(wrapper) == expected
    assert train.optimizer_evidence(wrapper.optimizer) == expected
    assert all(group['params'] == [] for group in adam.param_groups)
    saved = tmp_path / 'optimizer.pt'
    torch.save(wrapper.state_dict(), saved)
    for key in ('exp_avg', 'exp_avg_sq', 'step'):
        value = next(iter(adam.state.values()))[key]
        before = value.clone()
        value.add_(1.)
        assert train.optimizer_evidence(wrapper)['optimizer_state_sha256'] != expected['optimizer_state_sha256']
        value.copy_(before)
    adam.param_groups[0]['lr'] *= 2
    assert train.optimizer_evidence(wrapper)['optimizer_state_sha256'] != expected['optimizer_state_sha256']
    zero = wrapper.optimizer
    zero._set_fp32_optimizer_param_groups()
    adam.load_state_dict(torch.load(saved, map_location='cpu', weights_only=False)['optimizer_state_dict'])
    zero._clear_fp32_optimizer_param_groups()
    assert train.optimizer_evidence(wrapper) == expected
    assert all(group['params'] == [] for group in adam.param_groups)


def test_checkpoint_config_and_world_mismatch_rejected(tmp_path):
    model = TinyPlanner()
    opt, scheduler = make_optimizer(model)
    checkpoint = train.save_checkpoint(tmp_path / 'ckpt', model, opt, scheduler,
        dict(epoch=0, position=0, update=0, source_exposures=0), 'fixture')
    with pytest.raises(ValueError, match='Strict checkpoint'):
        train.load_checkpoint(checkpoint, model, opt, scheduler, 'other')
    with pytest.raises(ValueError, match='Strict checkpoint'):
        train.load_checkpoint(checkpoint, model, opt, scheduler, 'fixture', world=8)


def test_precision_groups_reject_bf16_trainables():
    model = TinyPlanner()
    model.lora_A.data = model.lora_A.data.to(torch.bfloat16)
    with pytest.raises(ValueError, match='FP32'):
        train.optimizer_groups(model, 1e-5, 5e-5)


def test_zero_partition_cannot_hide_bf16_lora_under_fp32_placeholder():
    model = TinyPlanner()
    model.lora_A.ds_tensor = torch.ones(1, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match='ZeRO partition dtype'):
        train.optimizer_groups(model, 1e-5, 5e-5)


def test_unexpected_unfrozen_base_is_not_silently_a_projector():
    model = TinyPlanner()
    model.language_decoder = torch.nn.Parameter(torch.ones(3))
    with pytest.raises(ValueError, match='Unexpected unfrozen base'):
        train.optimizer_groups(model, 1e-5, 5e-5)


def test_zero3_config_preserves_fp32_lora_and_actual_batch():
    args = train.parser().parse_args(['--family', 'moss', '--model', '.', '--train', 'a', '--val', 'b', '--out', 'c'])
    config = train.deepspeed_config(args, 8)
    assert config['train_batch_size'] == 32
    assert config['zero_optimization']['stage'] == 3
    assert config['bf16']['enabled'] is False and config['fp16']['enabled'] is False
    assert config['torch_autocast'] == dict(enabled=True, dtype='bfloat16')
    assert config['communication_data_type'] == 'fp32'


def test_ds_same_process_resume_is_explicitly_rejected(tmp_path):
    model = TinyPlanner()
    opt, scheduler = make_optimizer(model)
    checkpoint = train.save_checkpoint(tmp_path / 'ckpt', model, opt, scheduler,
        dict(epoch=0, position=0, update=0, source_exposures=0), 'fixture')
    marker_path = checkpoint / 'trainer_state.json'
    marker = json.loads(marker_path.read_text())
    marker['zero3'] = True
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='fresh process/engine'):
        train.load_checkpoint(checkpoint, model, opt, scheduler, 'fixture', zero3=True)


def test_ds_loaded_weights_are_compared_not_overwritten(tmp_path, monkeypatch):
    class FakeDSEngine(TinyPlanner):
        def load_checkpoint(self, *args, **kwargs):
            assert kwargs['load_optimizer_states'] is True
            assert kwargs['load_module_strict'] is False
            assert kwargs['custom_load_fn'](src={}, dst=self) is None
            return 'loaded', {}
    model = FakeDSEngine()
    opt, scheduler = make_optimizer(model)
    checkpoint = train.save_checkpoint(tmp_path / 'ckpt', model, opt, scheduler,
        dict(epoch=0, position=0, update=0, source_exposures=0), 'fixture')
    marker_path = checkpoint / 'trainer_state.json'
    marker = json.loads(marker_path.read_text())
    marker.update(zero3=True, files=[])
    marker_path.write_text(json.dumps(marker))
    writer_path = checkpoint / 'writer-rank0.json'
    writer = json.loads(writer_path.read_text())
    writer['nonce'] = 'different-process-fixture'
    writer_path.write_text(json.dumps(writer))
    corrupted = train.trainable_state(model)
    corrupted['lora_A'].add_(1)
    monkeypatch.setattr(train, 'trainable_state', lambda *args: corrupted)
    monkeypatch.setattr(train, 'restore_trainable', lambda *args: pytest.fail('Do not overwrite a DS mismatch'))
    with pytest.raises(ValueError, match='restored trainable bytes differ'):
        train.load_checkpoint(checkpoint, model, opt, scheduler, 'fixture', zero3=True)


def test_outputs_smoke_does_not_claim_resume_and_requires_actual_gradients(tmp_path):
    model = TinyPlanner()
    opt, scheduler = make_optimizer(model)
    update(model, opt, scheduler)
    update(model, opt, scheduler)
    state = dict(epoch=0, position=64, update=2, source_exposures=64, best_val_nll=None)
    checkpoint = train.save_checkpoint(tmp_path / 'checkpoint-0000002', model, opt, scheduler, state, 'fixture')
    train.atomic_json(tmp_path / 'run_config.json', dict(fingerprint='fixture'))
    evidence_path = tmp_path / 'backward-evidence-u0000000-rank0.json'
    evidence = dict(starting_update=0, actual_backward_observed=True, actual_checkpoint_calls=dict(decoder=2),
                    gradient_norms=dict(lora_A=1., audio_projector=2.))
    train.atomic_json(evidence_path, evidence)
    args = train.parser().parse_args(['--family', 'moss', '--model', '.', '--train', 'a', '--val', 'b',
                                    '--out', str(tmp_path), '--max-updates', '2'])
    data = dict(train=dict(sources=10720, sha256='a' * 64), val=dict(sources=490, sha256='b' * 64))
    result = train.training_outputs(args, state, data, checkpoint, 'fixture', 0, 1)
    assert result['schema'] == 'music-planner-outputs-v1'
    assert result['mode'] == 'smoke'
    assert result['optimizer_updates'] == result['invocation_optimizer_updates'] == 2
    assert result['resume_verified'] is result['verified_new_process'] is False
    assert result['source_exposure_budget'] == 21440
    assert result['quality_accepted'] is False
    assert all(value['nonzero'] and value['finite'] for value in result['gradient_groups'].values())
    assert result['checkpoint']['sha256'] == train.file_sha(checkpoint / 'trainer_state.json')
    evidence['gradient_norms']['audio_projector'] = 0.
    train.atomic_json(evidence_path, evidence)
    with pytest.raises(ValueError, match='nonzero language LoRA/audio bridge'):
        train.training_outputs(args, state, data, checkpoint, 'fixture', 0, 1)




@pytest.mark.parametrize('precision', ['bf16', 'fp16'])
def test_native_autocast_config_rejects_trainable_downcast(tmp_path, precision):
    args = train.parser().parse_args(['--family', 'moss', '--model', '.', '--train', 'a', '--val', 'b', '--out', 'c'])
    args.deepspeed = tmp_path / 'ds.json'
    args.deepspeed.write_text(json.dumps({precision: dict(enabled=True)}))
    with pytest.raises(ValueError, match='downcast FP32 LoRA'):
        train.deepspeed_config(args, 8)


def test_native_autocast_config_has_real_deepspeed_parser_contract():
    pytest.importorskip('deepspeed')
    from deepspeed.runtime.config import DeepSpeedConfig
    args = train.parser().parse_args(['--family', 'moss', '--model', '.', '--train', 'a', '--val', 'b', '--out', 'c'])
    parsed = DeepSpeedConfig(train.deepspeed_config(args, 1))
    assert parsed.torch_autocast_enabled is True
    assert parsed.torch_autocast_dtype == torch.bfloat16
    assert parsed.communication_data_type == torch.float32
    assert parsed.bfloat16_config.enabled is False
    assert parsed.float16_config.enabled is False
    assert parsed.zero_config.stage == 3


@pytest.mark.parametrize('name', ['base', 'lora_A'])
@pytest.mark.parametrize('attribute', ['ds_tensor', 'ds_secondary_tensor'])
def test_native_storage_contract_checks_both_real_zero_partitions(name, attribute):
    model = TinyPlanner()
    model.base.data = model.base.data.float()
    setattr(getattr(model, name), attribute, torch.ones(1, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match=attribute):
        train.optimizer_groups(model, 1e-5, 5e-5, base_dtype=torch.float32)


def test_native_storage_preparation_preserves_ddp_bf16_base():
    model = TinyPlanner()
    before = copy.deepcopy(model.state_dict())
    base_dtype = train.prepare_training_storage(model, None, False)
    assert base_dtype == torch.bfloat16
    assert_nested_equal(model.state_dict(), before)
    train.optimizer_groups(model, 1e-5, 5e-5, base_dtype=base_dtype)


@pytest.fixture
def native_ds_engine(monkeypatch):
    pytest.importorskip('deepspeed')
    from deepspeed.runtime.config import DeepSpeedConfig
    from deepspeed.runtime.engine import DeepSpeedEngine
    from deepspeed.runtime import torch_autocast
    monkeypatch.setattr(torch_autocast, 'TORCH_AUTOCAST_INITIALIZED', torch_autocast.TORCH_AUTOCAST_INITIALIZED)
    args = train.parser().parse_args(['--family', 'moss', '--model', '.', '--train', 'a', '--val', 'b', '--out', 'c'])
    parsed = DeepSpeedConfig(train.deepspeed_config(args, 1))

    class ContractEngine(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.module = TinyPlanner()
            self.module.base_linear = torch.nn.Linear(3, 3, dtype=torch.bfloat16).requires_grad_(False)
            self.communication_data_type = parsed.communication_data_type

        def fp16_enabled(self):
            return parsed.float16_config.enabled

        def bfloat16_enabled(self):
            return parsed.bfloat16_config.enabled

        def zero_quantized_weights(self):
            return parsed.zero_config.zero_quantized_weights

        def torch_autocast_enabled(self):
            return parsed.torch_autocast_enabled

        def torch_autocast_dtype(self):
            return parsed.torch_autocast_dtype

        def autotuning_profile_model_info(self):
            return False

        def is_deepcompile_enabled(self):
            return False

        def forward(self, **kwargs):
            assert not torch.is_autocast_enabled('cpu')
            return SimpleNamespace(logits=DeepSpeedEngine.forward(self, kwargs['x']))

    return ContractEngine(), parsed, torch_autocast


def test_real_ds_autocast_rejects_mixed_storage_then_accepts_exact_fp32_upcast(native_ds_engine):
    from scripts_omnidance.exp.ms.music_planner_adapters import load_planner_adapter
    engine, parsed, autocast = native_ds_engine
    model = engine.module
    for parameter in model.parameters():
        parameter.ds_tensor = parameter.detach().flatten()[:2].clone()
        parameter.ds_secondary_tensor = parameter.detach().flatten()[:1].clone()
        parameter.ds_tensor.storage_identity = 'primary-preserved'
        parameter.ds_secondary_tensor.storage_identity = 'secondary-preserved'
    before = {name: (p.detach().clone(), p.requires_grad, p.ds_tensor, p.ds_tensor.clone(),
                     p.ds_secondary_tensor, p.ds_secondary_tensor.clone()) for name, p in model.named_parameters()}
    with pytest.raises(AssertionError, match='All parameters must be float32'):
        autocast.init_autocast_params(engine, parsed.torch_autocast_dtype, parsed.torch_autocast_lower_precision_safe_modules)
    base_dtype = train.prepare_training_storage(model, load_planner_adapter, True)
    assert base_dtype == torch.float32
    for name, parameter in model.named_parameters():
        value, requires_grad, partition, partition_value, secondary, secondary_value = before[name]
        assert parameter.dtype == torch.float32 and parameter.requires_grad == requires_grad
        torch.testing.assert_close(parameter, value.float(), rtol=0, atol=0)
        assert parameter.ds_tensor is partition and parameter.ds_secondary_tensor is secondary
        assert partition.storage_identity == 'primary-preserved'
        assert secondary.storage_identity == 'secondary-preserved'
        assert partition.dtype == secondary.dtype == torch.float32
        torch.testing.assert_close(partition, partition_value.float(), rtol=0, atol=0)
        torch.testing.assert_close(secondary, secondary_value.float(), rtol=0, atol=0)
    train.optimizer_groups(model, 1e-5, 5e-5, base_dtype=base_dtype)
    autocast.init_autocast_params(engine, parsed.torch_autocast_dtype, parsed.torch_autocast_lower_precision_safe_modules)
    weight = model.base_linear.weight
    assert weight.dtype == torch.float32 and weight.ds_tensor.dtype == torch.float32
    assert autocast.get_autocast_dtype(weight) == torch.bfloat16
    output = engine(x=torch.randn(1, 4, 3))
    assert output.logits.dtype == torch.bfloat16
    loss, count = train.shifted_nll(output.logits, torch.tensor([[-100, 0, 1, 2]]))
    assert loss.dtype == torch.float32 and count.item() == 3
    optimizer = torch.optim.AdamW(train.optimizer_groups(model, 1e-5, 5e-5, base_dtype=base_dtype)[0])
    loss.backward()
    optimizer.step()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert train.optimizer_evidence(optimizer)['moments_fp32'] is True


@pytest.mark.parametrize('native', [False, True])
def test_validation_autocast_uses_native_engine_only_when_configured(tmp_path, native, native_ds_engine):
    from scripts_omnidance.exp.ms.music_planner_adapters import load_planner_adapter
    engine, parsed, autocast = native_ds_engine
    example = row(tmp_path, 'validation', 'val')
    labels = torch.tensor([[-100, 0, 1, 2]])
    batch = dict(x=torch.ones(1, 4, 3), labels=labels)
    adapter = SimpleNamespace(collate=lambda rows: {key: value.clone() for key, value in batch.items()})
    accelerator = SimpleNamespace(device=torch.device('cpu'), num_processes=1, process_index=0,
                                  reduce=lambda value, reduction: value)
    if native:
        train.prepare_training_storage(engine.module, load_planner_adapter, True)
        autocast.init_autocast_params(engine, parsed.torch_autocast_dtype, parsed.torch_autocast_lower_precision_safe_modules)
        model = engine
    else:
        class DDPFixture(TinyPlanner):
            def forward(self, x):
                assert torch.is_autocast_enabled('cpu')
                return SimpleNamespace(logits=super().forward(x))
        model = DDPFixture()
    result = train.evaluate(model, adapter, [example], train.AudioIdentityCache(tmp_path / 'cache.json'),
                            accelerator, 1, native_autocast=native)
    assert result['assistant_tokens'] == 3 and result['source_count'] == 1
    assert model.training is True and result['test_used'] is False


def test_missing_adapter_loader_is_explicit_blocked(monkeypatch):
    class Missing:
        pass
    monkeypatch.setattr(train.importlib, 'import_module', lambda _: Missing())
    with pytest.raises(RuntimeError, match='BLOCKED'):
        train.resolve_adapter()
