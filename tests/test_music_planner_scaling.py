"""CPU numerical and real two-process DDP checks for the multi-node trainer.

These fixtures validate loss/control contracts, not A800/ZeRO3 native-model runs.
"""
from contextlib import nullcontext
from datetime import timedelta
import importlib.util
import json
import os
import shutil
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from scripts_omnidance.exp.ms import music_planner_token_loss as losses
from scripts_omnidance.exp.ms import music_planner_train as train


@pytest.mark.parametrize('chunk', [0, 1, 3, 128])
def test_assistant_chunks_match_dense_value_and_gradient(chunk):
    torch.manual_seed(8)
    x = torch.randn(2, 8, 19, requires_grad=True)
    labels = torch.tensor([[-100, -100, 3, 2, 1, -100, 7, 4], [-100, 5, 1, -100, -100, 9, 2, 6]])
    expected = F.cross_entropy(x[:, :-1].reshape(-1, 19), labels[:, 1:].reshape(-1), ignore_index=-100, reduction='sum')
    expected_grad, = torch.autograd.grad(expected, x)
    actual, count = losses.assistant_token_nll(x, labels, chunk)
    actual_grad, = torch.autograd.grad(actual, x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert count.item() == 10
    assert not actual_grad[:, -1].any()
    assert not actual_grad[:, 0][labels[:, 1].eq(-100)].any()


@pytest.mark.parametrize('chunk', [0, 1, 128])
def test_empty_tail_has_finite_zero_gradient_without_reading_logits(chunk):
    logits = torch.full((1, 3, 5), float('nan'), requires_grad=True)
    total, count = losses.assistant_token_nll(logits, torch.full((1, 3), -100, dtype=torch.long), chunk)
    assert total.item() == 0 and count.item() == 0
    total.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_no_full_prefix_vocabulary_upcast(monkeypatch):
    calls = []
    original = losses.F.cross_entropy

    def record(input, target, **kwargs):
        calls.append(input.shape)
        return original(input, target, **kwargs)

    monkeypatch.setattr(losses.F, 'cross_entropy', record)
    labels = torch.full((1, 50), -100, dtype=torch.long)
    labels[0, -5:] = torch.tensor([1, 2, 3, 4, 5])
    total, count = losses.assistant_token_nll(torch.randn(1, 50, 11), labels, 2)
    assert count.item() == 5 and total.isfinite()
    assert calls == [torch.Size([2, 11]), torch.Size([2, 11]), torch.Size([1, 11])]


def test_independent_global_denominators_and_unknown_head_graph():
    caption = torch.tensor(21., requires_grad=True)
    head = torch.tensor(10., requires_grad=True)
    objective = losses.joint_update_objective(caption, head, torch.tensor(7.), torch.tensor(2.),
        world=64, accum=2, genre_weight=.3)
    torch.testing.assert_close(objective, torch.tensor((3. + .3 * 5.) * 128))
    objective.backward()
    torch.testing.assert_close(caption.grad, torch.tensor(128. / 7))
    torch.testing.assert_close(head.grad, torch.tensor(.3 * 128 / 2))
    head = torch.tensor(0., requires_grad=True)
    result = losses.joint_update_objective(caption, head, torch.tensor(7.), torch.tensor(0.),
        world=32, accum=1, genre_weight=.3)
    result.backward()
    assert head.grad is not None
    zero = torch.tensor(0., requires_grad=True)
    losses.joint_update_objective(caption, zero, torch.tensor(7.), torch.tensor(0.),
        world=32, accum=1, genre_weight=0.).backward()
    assert zero.grad is not None and zero.grad.item() == 0


@pytest.mark.parametrize('world', [32, 64])
@pytest.mark.parametrize('size', [1, 33, 65, 10713])
def test_exact_once_tail_and_resume_cursor_at_cluster_sizes(world, size):
    order = train.epoch_order(size, 42, 0)
    seen, position, updates = [], 0, 0
    while position < size:
        ranks = [train.update_slices(order, position, 1, 1, rank, world) for rank in range(world)]
        for slices, count in ranks:
            assert len(slices) == 1 and count == ranks[0][1]
            seen.extend(slices[0])
        position += ranks[0][1]
        updates += 1
        state = dict(epoch=int(position == size), position=0 if position == size else position,
            update=updates, source_exposures=position)
        train.validate_cursor(state, size, 2, world)
    assert sorted(seen) == list(range(size)) and len(seen) == len(set(seen))


def test_dummy_genre_never_repeats_borrowed_label():
    class Adapter:
        jev_classes = ['a', 'b']

        def collate(self, rows):
            return dict(labels=torch.tensor([[-100, 1, 2]]), jev_prefix_end=torch.tensor([1]))

    row = dict(jev_target=[1., 0.])
    cache = SimpleNamespace(verify=lambda _: None)
    batch = train.prepare_batch(Adapter(), [row], cache, empty=True)
    assert not batch['labels'].ne(-100).any()
    assert not batch['jev_label_mask'].any() and not batch['jev_targets'].any()
    moved = train.move_batch(batch, 'cpu')
    assert 'jev_prefix_end' in moved and 'jev_targets' not in moved and 'jev_label_mask' not in moved


def test_head_optimizer_group_is_fp32_separate_and_never_base():
    model = nn.Module()
    model.lora_A = nn.Parameter(torch.ones(2))
    model.audio_projector = nn.Linear(2, 2)
    model.jev_head = nn.Linear(2, 3)
    model.base = nn.Parameter(torch.ones(2, dtype=torch.bfloat16), requires_grad=False)
    groups, names = train.optimizer_groups(model, 1e-5, 2e-5, genre_lr=3e-5)
    assert {g['name']: g['lr'] for g in groups} == dict(lora=1e-5, projector=2e-5, jev=3e-5)
    assert set(names['jev']) == {'jev_head.weight', 'jev_head.bias'}


def test_moe_policy_explicitly_marks_native_captioner_blocks(monkeypatch):
    block = type('Qwen3OmniMoeThinkerTextSparseMoeBlock', (nn.Module,), {})
    model = nn.Sequential(block(), block())

    def mark(model, classes):
        for child in model.modules():
            if type(child) in classes:
                child._z3_leaf = True

    monkeypatch.setitem(sys.modules, 'deepspeed', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'deepspeed.utils', SimpleNamespace(set_z3_leaf_modules=mark))
    records = train.configure_zero3_leaf_modules(model, 'omni_captioner', True)
    assert len(records) == 2 and all(child._z3_leaf for child in model)
    assert train.configure_zero3_leaf_modules(model, 'moss', True) == []
    with pytest.raises(ValueError, match='not found'):
        train.configure_zero3_leaf_modules(nn.Linear(2, 2), 'omni_captioner', True)


def test_cluster_binding_and_constant_schedule_compatibility(monkeypatch):
    monkeypatch.delenv('PLANNER_CLUSTER_PLAN_SHA256', raising=False)
    assert train.cluster_environment(1) is None
    monkeypatch.setenv('PLANNER_CLUSTER_PLAN_SHA256', 'a' * 64)
    monkeypatch.setenv('PLANNER_EXPECTED_WORLD_SIZE', '32')
    monkeypatch.setenv('LOCAL_WORLD_SIZE', '8')
    monkeypatch.setenv('PLANNER_EXPECTED_HOSTNAME', train.socket.gethostname())
    monkeypatch.setenv('PLANNER_BASE_MODEL_INVENTORY_SHA256', 'b' * 64)
    assert train.cluster_environment(32)['world_size'] == 32
    with pytest.raises(ValueError, match='differs'):
        train.cluster_environment(64)
    for step in range(30):
        assert train.lr_multiplier(step, 30, 10) == min(1., (step + 1) / 10)
    assert train.lr_multiplier(29, 30, 3, 'cosine') == 0
    assert train.lr_multiplier(2, 30, 3, 'cosine') == 1


def test_actual_content_proof_accepts_mtime_changes_but_rejects_same_size_edit(tmp_path):
    from scripts_omnidance.exp.ms.music_planner_cluster import model_inventory
    (tmp_path / 'config.json').write_text('{}', encoding='utf-8')
    weight = tmp_path / 'model.safetensors'
    weight.write_bytes(b'first-weight-bytes')
    original = train.local_model_identity(tmp_path)
    inventory = model_inventory(tmp_path)['inventory_sha256']
    proof = [dict(inventory_sha256=inventory)] * 4
    canonical = train.canonical_cluster_model_identity(original, proof, inventory, 4)
    stat = weight.stat()
    os.utime(weight, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
    changed_mtime = train.local_model_identity(tmp_path)
    assert changed_mtime['weights'] != original['weights']
    assert model_inventory(tmp_path)['inventory_sha256'] == inventory
    assert train.canonical_cluster_model_identity(changed_mtime, proof, inventory, 4) == canonical
    weight.write_bytes(b'other-weight-bytes')
    assert len(weight.read_bytes()) == original['weights'][0]['size']
    actual = model_inventory(tmp_path)['inventory_sha256']
    with pytest.raises(ValueError, match='content verification'):
        train.canonical_cluster_model_identity(changed_mtime, [dict(inventory_sha256=actual)] * 4, inventory, 4)
    with pytest.raises(ValueError, match='content verification'):
        train.canonical_cluster_model_identity(changed_mtime, [dict(error='cannot read model')], inventory, 4)


class _JointFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4, 6)
        self.caption = nn.Linear(6, 11)
        self.genre = nn.Linear(6, 3)

    def forward(self, x):
        hidden = self.shared(x).tanh()
        return self.caption(hidden), self.genre(hidden[:, 0])


def _source(index):
    size = [3, 5, 4][index]
    x = torch.arange(size * 4, dtype=torch.float32).reshape(1, size, 4) / 20 + index
    labels = torch.full((1, size), -100, dtype=torch.long)
    labels[0, -[1, 3, 2][index]:] = torch.arange([1, 3, 2][index]) + 1
    return x, labels


def _ddp_worker(rank, store_path, output_dir):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method=Path(store_path).as_uri(),
        rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(16)
        model = torch.nn.parallel.DistributedDataParallel(_JointFixture())
        slices, _ = train.update_slices([0, 1, 2], 0, 1, 2, rank, 2)
        for step, indices in enumerate(slices):
            with model.no_sync() if step == 0 else nullcontext():
                index = indices[0] if indices else 0
                x, labels = _source(index)
                if not indices:
                    labels.fill_(-100)
                logits, genre_logits = model(x)
                caption, _ = losses.assistant_token_nll(logits, labels, 1)
                known = bool(indices) and index in (0, 2)
                genre = F.cross_entropy(genre_logits, torch.tensor([index]), reduction='sum') if known else genre_logits.reshape(-1)[:0].sum()
                loss = losses.joint_update_objective(caption, genre, torch.tensor(6.), torch.tensor(2.),
                    world=2, accum=2, genre_weight=.4)
                (loss / 2).backward()  # Accelerator/managed DS applies 1/GAS.
        torch.save({name: p.grad for name, p in model.module.named_parameters()},
                   Path(output_dir) / ('rank' + str(rank) + '.pt'))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
                    reason='CPU torch build has no Gloo backend')
def test_real_two_rank_ddp_matches_single_global_ragged_and_genre_objective(tmp_path):
    import torch.multiprocessing as mp
    try:
        torch.distributed.ProcessGroupGloo.create_device(hostname='127.0.0.1')
    except RuntimeError as error:
        if 'unsupported gloo device' in str(error):
            pytest.skip('Installed CPU torch advertises Gloo but has no usable transport device')
        raise
    try:
        mp.spawn(_ddp_worker, args=(str(tmp_path / 'store'), str(tmp_path)), nprocs=2, join=True)
    except mp.ProcessRaisedException as error:
        if 'unsupported gloo device' in str(error):
            pytest.skip('Installed CPU torch advertises Gloo but has no usable transport device')
        raise
    torch.manual_seed(16)
    model = _JointFixture()
    objective = 0
    for index in range(3):
        x, labels = _source(index)
        logits, genre_logits = model(x)
        caption, _ = losses.assistant_token_nll(logits, labels, 1)
        objective = objective + caption / 6
        if index in (0, 2):
            objective = objective + .4 * F.cross_entropy(genre_logits, torch.tensor([index])) / 2
    objective.backward()
    for rank in range(2):
        actual = torch.load(tmp_path / ('rank' + str(rank) + '.pt'), weights_only=True)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(actual[name], parameter.grad, rtol=2e-5, atol=1e-7)


def test_rank_averaged_accumulated_gradients_match_global_objective_without_transport():
    torch.manual_seed(16)
    reference = _JointFixture()
    expected_loss = 0
    for index in range(3):
        x, labels = _source(index)
        logits, genre_logits = reference(x)
        caption, _ = losses.assistant_token_nll(logits, labels, 1)
        expected_loss = expected_loss + caption / 6
        if index in (0, 2):
            expected_loss = expected_loss + .4 * F.cross_entropy(genre_logits, torch.tensor([index])) / 2
    expected_loss.backward()
    grads = []
    for rank in range(2):
        torch.manual_seed(16)
        model = _JointFixture()
        slices, _ = train.update_slices([0, 1, 2], 0, 1, 2, rank, 2)
        for indices in slices:
            index = indices[0] if indices else 0
            x, labels = _source(index)
            if not indices:
                labels.fill_(-100)
            logits, genre_logits = model(x)
            caption, _ = losses.assistant_token_nll(logits, labels, 1)
            genre = (F.cross_entropy(genre_logits, torch.tensor([index]), reduction='sum')
                     if indices and index in (0, 2) else genre_logits.reshape(-1)[:0].sum())
            objective = losses.joint_update_objective(caption, genre, torch.tensor(6.), torch.tensor(2.),
                world=2, accum=2, genre_weight=.4)
            (objective / 2).backward()
        grads.append({name: p.grad for name, p in model.named_parameters()})
    for name, parameter in reference.named_parameters():
        torch.testing.assert_close((grads[0][name] + grads[1][name]) / 2, parameter.grad, rtol=2e-5, atol=1e-7)


class _TrainingDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=3)
        self.lora_A = nn.Parameter(torch.randn(3, 3) * .1)
        self.audio_projector = nn.Linear(3, 3)
        self.norm = nn.LayerNorm(3, dtype=torch.bfloat16)
        self.norm.requires_grad_(False)
        self.gradient_checkpointing = False

    def forward(self, values):
        def body(x):
            return (self.audio_projector(x) @ self.lora_A).tanh()
        hidden = self._gradient_checkpointing_func(body, values) if self.gradient_checkpointing and self.training else body(values)
        return self.norm(hidden)


class _TrainingNative(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True, language_config=SimpleNamespace(hidden_size=3))
        self.language_model = _TrainingDecoder()
        self.embedding = nn.Embedding(7, 3, dtype=torch.bfloat16).requires_grad_(False)
        self.lm_head = nn.Linear(3, 7, dtype=torch.bfloat16).requires_grad_(False)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        from torch.utils.checkpoint import checkpoint
        assert gradient_checkpointing_kwargs == {'use_reentrant': True}
        self.language_model.gradient_checkpointing = True
        self.language_model._gradient_checkpointing_func = lambda fn, x: checkpoint(fn, x, use_reentrant=True)

    def forward(self, input_ids, attention_mask=None):
        values = self.embedding(input_ids)
        if self.training:
            values.requires_grad_(True)
        return SimpleNamespace(logits=self.lm_head(self.language_model(values)))


class _TrainingAdapter:
    family = 'moss'

    def __init__(self):
        self.model = _TrainingNative()

    def collate(self, rows):
        return dict(input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]]),
            attention_mask=torch.ones(1, 6, dtype=torch.long),
            labels=torch.tensor([[-100, -100, -100, 4, 5, 6]]), jev_prefix_end=torch.tensor([3]))

    def save_pretrained(self, path, state_dict=None):
        Path(path).mkdir()
        torch.save(state_dict, Path(path) / 'fixture.pt')


def _training_fixture(tmp_path, monkeypatch, genre):
    pytest.importorskip('accelerate')
    schema = train.planner_module('music_planner_schema')
    rows = [dict(id='train-' + str(i), jev_target=None) for i in range(6)]
    order = train.epoch_order(6, 42, 0)
    rows[order[4]]['jev_target'] = [1., 0.]
    val = [dict(id='val', jev_target=None)]
    data = dict(train=dict(sources=6, sha256='a' * 64), val=dict(sources=1, sha256='b' * 64))
    if genre:
        data['jev'] = dict(classes=['hip-hop', '手势舞'])
    monkeypatch.setattr(train, 'check_data', lambda *args: (rows, val, data))
    monkeypatch.setattr(schema, 'read_training_data', lambda *args: (rows, val, data))
    monkeypatch.setattr(schema, 'configure_adapter', lambda adapter, args: adapter)
    monkeypatch.setattr(train, 'local_model_identity', lambda *args: dict(fixture=True))
    monkeypatch.setattr(train.AudioIdentityCache, 'verify', lambda *args: None)
    monkeypatch.setattr(train, 'resolve_adapter', lambda: (lambda **kwargs: _TrainingAdapter()))
    classes = tmp_path / 'classes.json'
    classes.write_text(json.dumps(['hip-hop', '手势舞'], ensure_ascii=False), encoding='utf-8')
    argv = ['--family', 'moss', '--model', '.', '--train', 'train', '--val', 'val', '--out',
        str(tmp_path / 'run'), '--backend', 'ddp', '--epochs', '1', '--batch', '2', '--accum', '2',
        '--eval-every', '1', '--save-every', '0', '--loss-chunk-size', '2']
    if genre:
        argv += ['--genre-classes', str(classes), '--genre-weight', '.3', '--genre-head-dim', '4']
    return train.parser().parse_args(argv)


def test_actual_trainer_interval_evaluation_selects_by_full_val_caption_nll(tmp_path, monkeypatch):
    # This adapter supplies an extra prefix field only in the JEV test.
    original = _TrainingAdapter.collate

    def caption_batch(self, rows):
        batch = original(self, rows)
        batch.pop('jev_prefix_end')
        return batch

    monkeypatch.setattr(_TrainingAdapter, 'collate', caption_batch)
    args = _training_fixture(tmp_path, monkeypatch, genre=False)
    train.train(args)
    assert (args.out / 'best.json').is_file()
    for update in (1, 2, 3):
        validation = json.loads((args.out / f'checkpoint-{update:07d}' / 'validation.json').read_text(encoding='utf-8'))
        assert validation['source_count'] == 1 and validation['assistant_tokens'] == 3
    output = json.loads((args.out / 'outputs.json').read_text(encoding='utf-8'))
    assert output['source_exposures'] == 6 and output['full_budget_completed'] is True


def test_sparse_jev_smoke_unknown_then_fresh_resume_known_saves_bound_head(tmp_path, monkeypatch):
    args = _training_fixture(tmp_path, monkeypatch, genre=True)
    args.max_updates = 2
    train.train(args)
    first = json.loads((args.out / 'outputs.json').read_text(encoding='utf-8'))
    assert first['jev']['status'] == 'head_not_exercised'
    assert first['jev']['known_label_source_exposures'] == 0
    assert not (args.out / 'best.json').exists()
    args.resume = args.out / 'checkpoint-0000002'
    args.max_updates = 3
    monkeypatch.setattr(train, 'PROCESS_IDENTITY', dict(train.PROCESS_IDENTITY, nonce='fresh-jev-process-fixture'))
    train.train(args)
    final = json.loads((args.out / 'outputs.json').read_text(encoding='utf-8'))
    assert final['jev']['status'] == 'head_backward_exercised'
    assert final['jev']['known_label_source_exposures'] == 1
    assert final['resumed_next_update_performed'] is True
    checkpoint = args.out / 'checkpoint-0000003'
    head = json.loads((checkpoint / 'jev' / 'metadata.json').read_text(encoding='utf-8'))
    marker = json.loads((checkpoint / 'trainer_state.json').read_text(encoding='utf-8'))
    assert head['installation']['classes'] == ['hip-hop', '手势舞']
    assert head['identity']['run_fingerprint'] == marker['fingerprint']
    assert head['identity']['trainable_sha256'] == marker['trainable_sha256']
    portable = torch.load(checkpoint / 'trainable.pt', weights_only=True)
    assert any(name.startswith('jev_head.') for name in portable)
    assert all(torch.isfinite(value).all() for value in portable.values())


def _saved_resume_fixture(tmp_path):
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    state = dict(epoch=0, position=1, update=1, source_exposures=1)
    path = tmp_path / 'original' / 'checkpoint-0000001'
    train.save_checkpoint(path, model, optimizer, scheduler, state, 'fixture')
    return path, model, optimizer, scheduler, state


def test_relocated_resume_checks_consumed_copy_not_original_paths(tmp_path, monkeypatch):
    original, model, optimizer, scheduler, state = _saved_resume_fixture(tmp_path)
    copied = tmp_path / 'copied' / 'renamed-checkpoint'
    shutil.copytree(original, copied)
    actual_sha = train.file_sha
    consumed = []

    def only_copy(path):
        resolved = Path(path).resolve()
        assert original.resolve() not in resolved.parents
        consumed.append(resolved)
        return actual_sha(path)

    monkeypatch.setattr(train, 'file_sha', only_copy)
    assert train.load_checkpoint(copied, model, optimizer, scheduler, 'fixture') == state
    assert (copied / 'scheduler.pt').resolve() in consumed
    # This was accepted when marker absolute paths checked the untouched original.
    changed = torch.load(copied / 'scheduler.pt', weights_only=True)
    changed['last_epoch'] = 999
    torch.save(changed, copied / 'scheduler.pt')
    with pytest.raises(ValueError, match='scheduler.pt'):
        train.load_checkpoint(copied, model, optimizer, scheduler, 'fixture')
    assert scheduler.last_epoch == 1


@pytest.mark.parametrize('member', ['scheduler.pt', 'rng-rank0.pt', 'optimizer-rank0.pt', 'writer-rank0.json'])
def test_resume_rejects_omitted_consumed_manifest_member(tmp_path, member):
    path, model, optimizer, scheduler, _ = _saved_resume_fixture(tmp_path)
    marker = json.loads((path / 'trainer_state.json').read_text(encoding='utf-8'))
    marker['files'] = [entry for entry in marker['files'] if Path(entry['path']).name != member]
    train.atomic_json(path / 'trainer_state.json', marker)
    with pytest.raises(ValueError, match='omits a consumed'):
        train.load_checkpoint(path, model, optimizer, scheduler, 'fixture')


def test_resume_rejects_duplicate_and_outside_recorded_members(tmp_path):
    path, _, _, _, _ = _saved_resume_fixture(tmp_path)
    marker = json.loads((path / 'trainer_state.json').read_text(encoding='utf-8'))
    scheduler = next(entry for entry in marker['files'] if Path(entry['path']).name == 'scheduler.pt')
    duplicate = dict(marker, files=marker['files'] + [dict(scheduler)])
    with pytest.raises(ValueError, match='Duplicate'):
        train.checkpoint_member_refs(path, duplicate)
    root = next(entry['path'].replace('\\', '/')[:-len('/trainable.pt')]
                for entry in marker['files'] if entry['path'].replace('\\', '/').endswith('/trainable.pt'))
    for relative in ('../outside.pt', 'adapter/../../outside.pt', './scheduler.pt'):
        escaped = dict(marker, files=marker['files'] + [dict(path=root + '/' + relative, sha256='0' * 64)])
        with pytest.raises(ValueError, match='escapes'):
            train.checkpoint_member_refs(path, escaped)
    outside = dict(marker, files=marker['files'] + [dict(path=root + '-foreign/scheduler.pt', sha256='0' * 64)])
    with pytest.raises(ValueError, match='outside its recorded root'):
        train.checkpoint_member_refs(path, outside)


def test_resume_manifest_windows_paths_resolve_inside_current_checkpoint(tmp_path):
    path, _, _, _, _ = _saved_resume_fixture(tmp_path)
    marker = json.loads((path / 'trainer_state.json').read_text(encoding='utf-8'))
    rewritten = []
    for entry in marker['files']:
        relative = Path(entry['path']).relative_to(path.resolve()).as_posix()
        rewritten.append(dict(entry, path='Z:\\old-machine\\saved-run\\checkpoint-0000001\\' + relative.replace('/', '\\')))
    refs = train.checkpoint_member_refs(path, dict(marker, files=rewritten))
    assert Path(refs['scheduler.pt']['consumed_path']) == (path / 'scheduler.pt').resolve()


def test_resume_rejects_symlink_escape(tmp_path):
    path, _, _, _, _ = _saved_resume_fixture(tmp_path)
    outside = tmp_path / 'outside.bin'
    outside.write_bytes(b'outside-checkpoint')
    link = path / 'escape.bin'
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip('This OS/account cannot create a file symlink')
    marker = json.loads((path / 'trainer_state.json').read_text(encoding='utf-8'))
    marker['files'].append(dict(path=str(link), sha256=train.file_sha(outside)))
    with pytest.raises(ValueError, match='symlink escapes'):
        train.checkpoint_member_refs(path, marker)


def test_manifest_failure_is_broadcast_before_rank_barrier(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'broadcast_object_list', lambda values, src: events.append(values[0]))
    with pytest.raises(ValueError, match='manifest is missing'):
        train.verify_checkpoint_members(tmp_path, {}, rank=0, world=2)
    assert len(events) == 1 and 'manifest is missing' in events[0]
