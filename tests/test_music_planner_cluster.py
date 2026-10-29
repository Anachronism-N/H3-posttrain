import json
import copy
import os
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest

from scripts_omnidance.exp.ms import music_planner_cluster as cluster


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    return str(path)


@pytest.fixture
def spec(tmp_path):
    repo = tmp_path / 'repo with spaces'
    scripts = repo / 'scripts_omnidance' / 'exp' / 'ms'
    scripts.mkdir(parents=True)
    # Real subprocess data-only receipt without importing any GPU/model library.
    (scripts / 'music_planner_train.py').write_text(
        'import json\nprint(json.dumps({"actual_training_started": False, "fixture": True}))\n')
    (scripts / 'music_planner_adapters.py').write_text('# fixture adapter\n')
    (scripts / 'music_planner_cluster.py').write_text('# fixture launcher\n')
    train = tmp_path / 'train.jsonl'
    train.write_text(json.dumps({'id': 'train-a', 'caption': '[0-2s] 动作。'}) + '\n' +
                     json.dumps({'id': 'train-b', 'caption': '[2-5s] 动作。'}) + '\n', encoding='utf-8')
    val = tmp_path / 'val.jsonl'
    val.write_text(json.dumps({'id': 'val-a'}) + '\n', encoding='utf-8')
    jobs = []
    for index, family in enumerate(cluster.FAMILIES):
        model = tmp_path / family
        dump(model / 'config.json', {'fixture': True})
        (model / 'model.safetensors').write_bytes(b'fixture weights')
        jobs.append(dict(family=family, hosts=['node%d' % i for i in range(index * 4, index * 4 + 4)],
                         model=str(model), train=str(train), val=str(val), out=str(tmp_path / ('out-' + family)),
                         master_port=29551 + index, job_id=family + '-new-run', repo_root=str(repo),
                         python=sys.executable, phase='smoke', max_updates=2))
    return dict(layout='split32', jobs=jobs)


def fake_runtime(version='fixture-1'):
    return dict(python_version='3.fixture', versions={name: version for name in cluster.DEPENDENCIES},
                torch_cuda='12.fixture', nccl_version=1, apis={name: True for name in cluster.APIS},
                gpus=[dict(local_rank=i, name='NVIDIA A800 80GB PCIe', total_memory=85_000_000_000)
                      for i in range(8)])


def native_check(job):
    return dict(actual_training_started=False, fixture=True)


def all_receipts(plan, directory):
    for job in plan['jobs']:
        for host in job['hosts']:
            cluster.preflight(plan, directory, hostname=host, runtime=fake_runtime, data_checker=native_check)


def test_split32_rank_mapping_flags_and_quoted_shell(spec):
    plan = cluster.build_plan(spec)
    assert plan['actual_training_started'] is False
    assert plan['gpu_runtime_verified'] is False
    commands = cluster.command_plan(plan)
    assert len(commands) == 8
    for command in commands:
        assert command['argv'][command['argv'].index('--nnodes') + 1] == '4'
        assert command['argv'][command['argv'].index('--nproc-per-node') + 1] == '8'
        assert command['argv'][command['argv'].index('--batch') + 1] == '32'
        assert command['argv'][command['argv'].index('--accum') + 1] == '1'
        assert command['argv'][command['argv'].index('--rdzv-backend') + 1] == 'static'
        assert command['argv'][command['argv'].index('--max-restarts') + 1] == '0'
        assert command['env']['PLANNER_EXPECTED_HOSTNAME'] == command['host']
        assert command['env']['PLANNER_CLUSTER_PLAN_SHA256'] == plan['training_contract_sha256']
        job = next(item for item in plan['jobs'] if item['job_id'] == command['job_id'])
        assert command['env']['PLANNER_BASE_MODEL_INVENTORY_SHA256'] == job['model']['inventory_sha256']
        shell = shlex.split(command['shell'])
        assert shell[0:3] == ['cd', command['cwd'], '&&']
        assert shell[-len(command['argv']):] == command['argv']
    assert sorted(rank for command in commands[:4] for rank in command['expected_ranks']) == list(range(32))


def test_single64_requires_eight_hosts_and_batch64(spec):
    spec['layout'] = 'single64'
    spec['jobs'] = spec['jobs'][:1]
    spec['jobs'][0]['hosts'] = ['node%d' % index for index in range(8)]
    plan = cluster.build_plan(spec)
    assert plan['jobs'][0]['world_size'] == 64
    assert plan['jobs'][0]['global_batch'] == 64
    assert len(cluster.command_plan(plan)) == 8
    spec['jobs'][0]['hosts'].pop()
    with pytest.raises(ValueError, match='hosts'):
        cluster.build_plan(spec)


@pytest.mark.parametrize('nodes,accum,batch', [(1, 4, 32), (2, 2, 32), (4, 1, 32), (8, 1, 64)])
def test_flexible_node_counts_and_accumulation(spec, nodes, accum, batch):
    spec['layout'] = 'flexible'
    spec['jobs'] = spec['jobs'][:1]
    spec['jobs'][0].update(hosts=['node%d' % index for index in range(nodes)], accum=accum)
    plan = cluster.build_plan(spec)
    job = plan['jobs'][0]
    assert (job['nodes'], job['world_size'], job['accum'], job['global_batch']) == (nodes, nodes * 8, accum, batch)
    commands = cluster.command_plan(plan)
    assert len(commands) == nodes
    for rank, command in enumerate(commands):
        argv = command['argv']
        assert argv[argv.index('--nnodes') + 1] == str(nodes)
        assert argv[argv.index('--batch') + 1] == str(batch)
        assert argv[argv.index('--accum') + 1] == str(accum)
        assert command['expected_ranks'] == list(range(rank * 8, rank * 8 + 8))


def test_flexible_eight_independent_single_node_experiments(spec):
    template = spec['jobs'][0]
    spec['layout'] = 'flexible'
    spec['jobs'] = []
    for index in range(8):
        job = copy.deepcopy(template)
        job.update(hosts=['node%d' % index], master_port=29551 + index,
                   job_id='experiment-%d' % index, out=template['out'] + '-%d' % index, accum=4)
        spec['jobs'].append(job)
    plan = cluster.build_plan(spec)
    assert len(plan['jobs']) == 8
    assert len(cluster.command_plan(plan)) == 8
    assert {job['global_batch'] for job in plan['jobs']} == {32}


def test_flexible_mixed_single_and_dual_node_preflight(spec, tmp_path):
    spec['layout'] = 'flexible'
    spec['jobs'][0].update(hosts=['node0'], accum=4)
    spec['jobs'][1].update(hosts=['node1', 'node2'], accum=2)
    plan = cluster.build_plan(spec)
    receipts = tmp_path / 'mixed-receipts'
    all_receipts(plan, receipts)
    admission = cluster.verify_receipts(plan, receipts)
    assert admission['hosts_checked'] == 3
    assert admission['local_preflight_passed']
    assert [job['world_size'] for job in plan['jobs']] == [8, 16]
    assert [job['global_batch'] for job in plan['jobs']] == [32, 32]


def test_flexible_resume_requires_original_world_size_even_at_same_batch(spec):
    spec.update(layout='flexible', jobs=spec['jobs'][:1])
    target = spec['jobs'][0]
    target.update(hosts=['node0'], accum=4)
    initial = cluster.build_plan(spec)
    checkpoint = Path(target['out']) / 'checkpoint-0000002'
    dump(checkpoint / 'trainer_state.json', {'complete': True, 'world_size': 8, 'cursor': {'update': 2}})
    dump(Path(target['out']) / 'run_config.json', {'fixture': True})
    target.update(resume=str(checkpoint), max_updates=3, job_id='resumed-single-node')
    resumed = cluster.build_plan(spec)
    assert initial['training_contract_sha256'] == resumed['training_contract_sha256']
    target.update(hosts=['node0', 'node1'], accum=2)
    with pytest.raises(ValueError, match='same world_size'):
        cluster.build_plan(spec)


@pytest.mark.parametrize('hosts,accum,error', [
    ([], 1, 'hosts'), (['a', 'b', 'c'], 1, 'hosts'), (['a'], 0, 'accum'),
    (['a'], True, 'accum'), (['a'], 1.5, 'accum'), (None, 1, 'hosts'),
])
def test_flexible_invalid_topology_or_accumulation(spec, hosts, accum, error):
    spec['layout'] = 'flexible'
    spec['jobs'] = spec['jobs'][:1]
    spec['jobs'][0].update(hosts=hosts, accum=accum)
    with pytest.raises(ValueError, match=error):
        cluster.build_plan(spec)


def test_flexible_limits_total_nodes_and_job_count(spec):
    spec['layout'] = 'flexible'
    spec['jobs'][0]['hosts'] = ['a%d' % index for index in range(8)]
    spec['jobs'][1]['hosts'] = ['b0']
    with pytest.raises(ValueError, match='at most eight nodes'):
        cluster.build_plan(spec)
    spec['jobs'] = []
    with pytest.raises(ValueError, match='one to eight jobs'):
        cluster.build_plan(spec)
    spec['jobs'] = [{}] * 9
    with pytest.raises(ValueError, match='one to eight jobs'):
        cluster.build_plan(spec)


def test_explicit_default_accum_keeps_legacy_contract_identity(spec):
    initial = cluster.build_plan(spec)
    for job in spec['jobs']:
        job['accum'] = 1
    explicit = cluster.build_plan(spec)
    assert explicit == initial


def test_flexible_accum_changes_training_fingerprint(spec):
    spec.update(layout='flexible', jobs=spec['jobs'][:1])
    spec['jobs'][0].update(hosts=['node0'], accum=4)
    first = cluster.build_plan(spec)
    spec['jobs'][0]['accum'] = 2
    assert cluster.build_plan(spec)['training_contract_sha256'] != first['training_contract_sha256']


def test_explicit_sft_retains_caption_vocabulary_without_jev_labels(spec, tmp_path):
    classes = dump(tmp_path / 'classes.json', ['contemporary', 'unknown'])
    for job in spec['jobs']:
        job.update(enable_jev=False, genre_classes=classes, genre_weight=0)
    plan = cluster.build_plan(spec)
    job = plan['jobs'][0]
    assert not job['genre_enabled']
    assert job['genre_vocabulary']['classes'] == ['contemporary', 'unknown']
    assert '--no-enable-jev' in job['train_argv']
    assert '--train-genre-labels' not in job['train_argv']
    spec['jobs'][0]['enable_jev'] = 'false'
    with pytest.raises(ValueError, match='boolean'):
        cluster.build_plan(spec)


def test_explicit_jev_binds_protocol_flag(spec, tmp_path):
    classes = dump(tmp_path / 'classes.json', ['contemporary', 'unknown'])
    labels = dump(tmp_path / 'labels.json', {'fixture': True})
    for job in spec['jobs']:
        job.update(enable_jev=True, genre_classes=classes, genre_weight=.1,
                   train_genre_labels=labels, val_genre_labels=labels)
    plan = cluster.build_plan(spec)
    assert '--enable-jev' in plan['jobs'][0]['train_argv']
    spec['jobs'][0]['genre_weight'] = 0
    with pytest.raises(ValueError, match='positive genre_weight'):
        cluster.build_plan(spec)


@pytest.mark.parametrize('values,error', [
    ({'fusion_lr': .0001}, 'require musicfm_cache'),
    ({'fusion_hidden_dim': 256}, 'require musicfm_cache'),
    ({'enable_jev': False, 'genre_weight': .1}, 'Disabled JEV'),
])
def test_new_encoder_or_jev_configuration_errors(spec, values, error):
    spec['jobs'][0].update(values)
    with pytest.raises(ValueError, match=error):
        cluster.build_plan(spec)


def test_musicfm_cache_identity_source_and_values_are_bound(spec, tmp_path, monkeypatch):
    index = dump(tmp_path / 'musicfm-index.json', {'fixture': True})
    current = {'cache_sha256': 'a' * 64}
    fake_module = SimpleNamespace(FeatureStore=lambda path: SimpleNamespace(identity=dict(current)))
    monkeypatch.setitem(sys.modules, 'scripts_omnidance.exp.ms.music_planner_musicfm', fake_module)
    spec.update(layout='flexible', jobs=spec['jobs'][:1])
    target = spec['jobs'][0]
    script_dir = Path(target['repo_root']) / 'scripts_omnidance' / 'exp' / 'ms'
    for name in ('music_planner_musicfm.py', 'music_planner_encoder_fusion.py'):
        (script_dir / name).write_text('# fixture encoder source\n')
    target.update(hosts=['node0'], accum=4, musicfm_cache=index, fusion_lr=.0001, fusion_hidden_dim=256)
    plan = cluster.build_plan(spec)
    job = plan['jobs'][0]
    assert job['musicfm_identity'] == current
    for flag in ('--musicfm-cache', '--fusion-lr', '--fusion-hidden-dim'):
        assert flag in job['train_argv']
    assert {Path(ref['path']).name for ref in job['code']} >= {
        'music_planner_musicfm.py', 'music_planner_encoder_fusion.py'}
    cluster.verify_files(job)
    current['cache_sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='cache identity differs'):
        cluster.verify_files(job)
    target['family'] = 'omni_captioner'
    with pytest.raises(ValueError, match='only for MOSS'):
        cluster.build_plan(spec)


def test_musicfm_preflight_hashes_feature_bytes_once_not_on_receipt_verification(spec, tmp_path, monkeypatch):
    checked = []

    class FakeStore:
        def __init__(self, path):
            self.identity = {'cache_sha256': 'a' * 64}

        def validate_rows(self, rows, verify_bytes=False):
            checked.append((sorted(row['id'] for row in rows), verify_bytes))

    monkeypatch.setitem(sys.modules, 'scripts_omnidance.exp.ms.music_planner_musicfm',
                        SimpleNamespace(FeatureStore=FakeStore))
    spec.update(layout='flexible', jobs=spec['jobs'][:1])
    target = spec['jobs'][0]
    script_dir = Path(target['repo_root']) / 'scripts_omnidance' / 'exp' / 'ms'
    for name in ('music_planner_musicfm.py', 'music_planner_encoder_fusion.py'):
        (script_dir / name).write_text('# fixture encoder source\n')
    target.update(hosts=['node0'], musicfm_cache=dump(tmp_path / 'index.json', {'fixture': True}))
    plan = cluster.build_plan(spec)
    assert checked == []
    receipts = tmp_path / 'cache-receipts'
    all_receipts(plan, receipts)
    assert checked == [(['train-a', 'train-b', 'val-a'], True)]
    cluster.verify_receipts(plan, receipts)
    cluster.command_plan(plan)
    assert len(checked) == 1


def test_model_inventory_canonical_relative_paths_and_no_mtime_identity(spec):
    model = Path(spec['jobs'][0]['model'])
    dump(model / 'Z' / 'nested.json', {'fixture': True})
    dump(model / 'a.json', {'fixture': True})
    before = cluster.model_inventory(model)
    assert [item['relative'] for item in before['files']] == sorted(item['relative'] for item in before['files'])
    assert 'Z/nested.json' in [item['relative'] for item in before['files']]
    os.utime(model / 'model.safetensors', (1_000_000, 1_000_000))
    assert cluster.model_inventory(model)['inventory_sha256'] == before['inventory_sha256']


@pytest.mark.parametrize('change,error', [
    (lambda s: s['jobs'][1]['hosts'].__setitem__(0, 'node0'), 'share'),
    (lambda s: s['jobs'][1].__setitem__('master_port', 29551), 'ports'),
    (lambda s: s['jobs'][1].__setitem__('job_id', s['jobs'][0]['job_id']), 'job_id'),
    (lambda s: s['jobs'][1].__setitem__('family', 'moss'), 'exactly MOSS'),
    (lambda s: s['jobs'][0].__setitem__('unknown_flag', 1), 'unknown'),
    (lambda s: s['jobs'][0].__setitem__('master_addr', 'other-node'), 'master_addr'),
    (lambda s: s['jobs'][0].__setitem__('genre_weight', float('nan')), 'finite'),
    (lambda s: s['jobs'][0].__setitem__('genre_weight', 1), 'JEV requires'),
    (lambda s: s['jobs'][0].__setitem__('phase', 'full'), 'full requires'),
])
def test_invalid_cluster_specs_fail(spec, change, error):
    change(spec)
    with pytest.raises(ValueError, match=error):
        cluster.build_plan(spec)


def test_manifest_primary_id_alias_conflict_and_overlap(spec):
    path = Path(spec['jobs'][0]['train'])
    path.write_text(json.dumps({'id': 'a', 'source_id': 'b'}))
    with pytest.raises(ValueError, match='disagree'):
        cluster.build_plan(spec)
    path.write_text(json.dumps({'id': 'val-a'}))
    with pytest.raises(ValueError, match='overlap'):
        cluster.build_plan(spec)
    path.write_text(json.dumps({'source_id': 'train-alias'}) + '\n')
    assert cluster.build_plan(spec)['jobs'][0]['train']['source_count'] == 1


def test_output_directory_overlap_and_model_mutation_rejected(spec):
    spec['jobs'][1]['out'] = str(Path(spec['jobs'][0]['out']) / 'nested')
    with pytest.raises(ValueError, match='contain each other'):
        cluster.build_plan(spec)
    spec['jobs'][1]['out'] = str(Path(spec['jobs'][1]['model']) / 'run')
    with pytest.raises(ValueError, match='immutable model'):
        cluster.build_plan(spec)


def test_genre_vocabulary_and_training_flags_are_bound(spec, tmp_path):
    classes = dump(tmp_path / 'classes.json', ['Popping', '手势舞'])
    labels = dump(tmp_path / 'labels.json', {'fixture': True})
    contract = dump(tmp_path / 'contract.json', {'fixture': True})
    for job in spec['jobs']:
        job.update(genre_classes=classes, train_genre_labels=labels, val_genre_labels=labels,
                   genre_weight=0.2, caption_contract=contract, audio_last_n=4, loss_chunk_size=128,
                   lr_schedule='cosine', warmup_ratio=0.05, genre_head_dim=256)
    plan = cluster.build_plan(spec)
    job = plan['jobs'][0]
    assert job['genre_enabled']
    assert job['genre_vocabulary']['classes_sha256'] == cluster.digest(['Popping', '手势舞'])
    assert '--genre-head-dim' in job['train_argv']
    assert '--caption-contract' in job['train_argv']
    assert '--lr-schedule' in job['train_argv']
    dump(Path(classes), ['a' * 81, 'Popping'])
    with pytest.raises(ValueError, match='genre_classes'):
        cluster.build_plan(spec)


def test_resume_phase_hash_is_stable_but_launch_hash_changes(spec):
    initial = cluster.build_plan(spec)
    for job in spec['jobs']:
        out = Path(job['out'])
        checkpoint = out / 'checkpoint-0000002'
        dump(checkpoint / 'trainer_state.json', {'complete': True, 'world_size': 32, 'cursor': {'update': 2}})
        dump(out / 'run_config.json', {'fixture': True})
        job.update(resume=str(checkpoint), max_updates=3, job_id=job['job_id'] + '-resume')
    resumed = cluster.build_plan(spec)
    assert resumed['training_contract_sha256'] == initial['training_contract_sha256']
    assert resumed['contract_sha256'] != initial['contract_sha256']
    for job in spec['jobs']:
        job.update(phase='full', max_updates=0)
    full = cluster.build_plan(spec)
    assert full['training_contract_sha256'] == initial['training_contract_sha256']
    assert '--max-updates' not in full['jobs'][0]['train_argv']
    spec['jobs'][0]['rank'] = 64
    assert cluster.build_plan(spec)['training_contract_sha256'] != initial['training_contract_sha256']


def test_resume_requires_complete_same_world(spec):
    job = spec['jobs'][0]
    checkpoint = Path(job['out']) / 'checkpoint-0000002'
    dump(checkpoint / 'trainer_state.json', {'complete': True, 'world_size': 8})
    dump(Path(job['out']) / 'run_config.json', {'fixture': True})
    job['resume'] = str(checkpoint)
    with pytest.raises(ValueError, match='same world_size'):
        cluster.build_plan(spec)


def test_resume_smoke_requires_an_actual_next_update(spec):
    job = spec['jobs'][0]
    checkpoint = Path(job['out']) / 'checkpoint-0000002'
    dump(checkpoint / 'trainer_state.json', {'complete': True, 'world_size': 32, 'cursor': {'update': 2}})
    dump(Path(job['out']) / 'run_config.json', {'fixture': True})
    job['resume'] = str(checkpoint)
    with pytest.raises(ValueError, match='exceed'):
        cluster.build_plan(spec)


def test_dryrun_receipt_never_admits_gpu_training(spec, tmp_path):
    plan = cluster.build_plan(spec)
    receipts = tmp_path / 'receipts'
    called = []
    receipt = cluster.preflight(plan, receipts, dry_run=True, hostname='node0',
                                runtime=lambda: called.append('GPU'), data_checker=native_check)
    assert not called
    assert receipt['runtime'] is None
    assert receipt['gpu_runtime_verified'] is False
    assert receipt['passed'] is False
    with pytest.raises(ValueError, match='dry-run'):
        cluster.verify_receipts(plan, receipts)
    with pytest.raises(ValueError, match='receipt exists'):
        cluster.preflight(plan, receipts, dry_run=True, hostname='node0', data_checker=native_check)


def test_preflight_verifies_all_hosts_shared_probes_and_software(spec, tmp_path):
    plan = cluster.build_plan(spec)
    receipts = tmp_path / 'receipts'
    all_receipts(plan, receipts)
    result = cluster.verify_receipts(plan, receipts)
    assert result['hosts_checked'] == 8
    assert result['local_preflight_passed']
    assert result['training_smoke_passed'] is False
    assert result['network_bandwidth_tested'] is False
    receipt = cluster.read_json(receipts / 'node1.json')
    receipt['runtime']['versions']['torch'] = 'different'
    receipt['receipt_sha256'] = cluster.digest({k: v for k, v in receipt.items() if k != 'receipt_sha256'})
    cluster.write_json(receipts / 'node1.json', receipt)
    with pytest.raises(ValueError, match='differs between nodes'):
        cluster.verify_receipts(plan, receipts)


def test_admission_rechecks_bytes_changed_after_preflight(spec, tmp_path):
    plan = cluster.build_plan(spec)
    receipts = tmp_path / 'receipts'
    all_receipts(plan, receipts)
    path = Path(spec['jobs'][0]['train'])
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='bytes changed'):
        cluster.verify_receipts(plan, receipts)


@pytest.mark.parametrize('target', ['model', 'code', 'manifest'])
def test_changed_payload_bytes_fail_preflight(spec, tmp_path, target):
    plan = cluster.build_plan(spec)
    if target == 'model':
        path = Path(spec['jobs'][0]['model']) / 'model.safetensors'
    elif target == 'code':
        path = Path(spec['jobs'][0]['repo_root']) / 'scripts_omnidance/exp/ms/music_planner_adapters.py'
    else:
        path = Path(spec['jobs'][0]['train'])
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='changed|differ'):
        cluster.preflight(plan, tmp_path / 'receipts', dry_run=True, hostname='node0', data_checker=native_check)


def test_wrong_host_and_gpu_target_rejected(spec, tmp_path):
    plan = cluster.build_plan(spec)
    with pytest.raises(ValueError, match='hostname'):
        cluster.preflight(plan, tmp_path / 'a', dry_run=True, hostname='not-allocated', data_checker=native_check)
    wrong = fake_runtime()
    wrong['gpus'][3]['name'] = 'A100'
    with pytest.raises(ValueError, match='A800'):
        cluster.preflight(plan, tmp_path / 'b', hostname='node0', runtime=lambda: wrong, data_checker=native_check)
    wrong['gpus'][3]['name'] = 'A800'
    wrong['gpus'][3]['total_memory'] = 40_000_000_000
    with pytest.raises(ValueError, match='80GB'):
        cluster.preflight(plan, tmp_path / 'c', hostname='node0', runtime=lambda: wrong, data_checker=native_check)


def test_preflight_checks_native_api_and_existing_out(spec, tmp_path):
    plan = cluster.build_plan(spec)
    wrong = fake_runtime()
    wrong['apis']['zero3_plugin'] = False
    with pytest.raises(ValueError, match='APIs'):
        cluster.preflight(plan, tmp_path / 'api', hostname='node0', runtime=lambda: wrong, data_checker=native_check)
    Path(spec['jobs'][0]['out']).mkdir()
    with pytest.raises(ValueError, match='out now exists'):
        cluster.preflight(plan, tmp_path / 'out', dry_run=True, hostname='node0', data_checker=native_check)


def test_real_caption_schema_primary_id_is_accepted(spec, tmp_path):
    from scripts_omnidance.exp.ms import music_planner_schema as schema
    contract = dict(schema=schema.SCHEMA, caption_version='fixture-version', task_instruction='描述编舞。',
                    audio_min_seconds=1, audio_max_seconds=30, required_markers=[], require_target_windows=False)
    contract_path = dump(tmp_path / 'contract-real-schema.json', contract)
    for split in ('train', 'val'):
        path = Path(spec['jobs'][0][split])
        row = dict(id=split + '-source', source_group_id=split + '-song', split=split,
                   audio_path=str(tmp_path / (split + '.wav')), audio_sha256=('a' if split == 'train' else 'b') * 64,
                   audio_duration_seconds=5.0625, audio_window_seconds=[0., 5.0625],
                   target_text='舞者抬起右臂，随后侧踏。', caption_version='fixture-version',
                   semantic_approval=False, human_review_completed=False)
        path.write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')
        rows, _ = schema.read_contract_split(path, split, schema.load_contract(contract_path))
        assert rows[0]['id'] == split + '-source'
    for job in spec['jobs']:
        job['caption_contract'] = contract_path
    plan = cluster.build_plan(spec)
    assert plan['jobs'][0]['train']['source_count'] == 1


def test_plan_cli_and_native_data_check_do_not_dispatch(spec, tmp_path, capsys):
    source = Path(dump(tmp_path / 'spec.json', spec))
    output = tmp_path / 'plan.json'
    cluster.main(['plan', '--spec', str(source), '--output', str(output)])
    built = json.loads(capsys.readouterr().out)
    assert built['gpu_runtime_verified'] is False and built['dispatch_performed'] is False
    plan = cluster.load_plan(output)
    assert cluster.native_data_check(plan['jobs'][0])['actual_training_started'] is False
    cluster.main(['commands', '--plan', str(output), '--format', 'json'])
    value = json.loads(capsys.readouterr().out)
    assert len(value['commands']) == 8 and value['dispatch_performed'] is False
    assert output.is_file() and not Path(spec['jobs'][0]['out']).exists()
    with pytest.raises(ValueError, match='already exists'):
        cluster.main(['plan', '--spec', str(source), '--output', str(output)])
    corrupted = cluster.read_json(output)
    corrupted['jobs'][0]['global_batch'] = 1
    cluster.write_json(output, corrupted)
    with pytest.raises(ValueError, match='hash mismatch'):
        cluster.load_plan(output)
