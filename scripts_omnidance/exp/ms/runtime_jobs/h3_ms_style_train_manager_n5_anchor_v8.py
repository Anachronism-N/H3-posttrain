import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
MS = ROOT / 'diff/logs_omnidance/exp/MS'
SOURCE = ROOT / 'H3-posttrain/scripts_omnidance/exp/ms/h3_video_sft.py'
WRAPPER = SOURCE.with_name('h3_video_only.py')
METADATA = MS / 'h3_ms_style_pilot_metadata_v1/h3_video_train.json'
CACHE = Path('/opt/h3cache/ms_pilot512_local_20261023_v1')
CACHE_QUEUE = MS / 'h3_ms_style_pilot_cache_queue_v2'
HELPER = MS / 'h3_ms_style_cache_manager_n5_v2.py'
HELPER_SHA = '337c6221c8b134235d901aa32759f216c0cc6fef750d08730a1ab7a028fe78fe'
SOURCE_SHA = '247e77eb365685a92680bd3934ac172c7d81242f06fb28f7f4aa665130233c63'
METADATA_SHA = '6b145dfb01e3468b50a0e1870639551c726662367744d2dc4d47014f54d3cacf'
WRAPPER_SHA = '89439d733d62f522bb3dd993bc456541b3ca15ead34ff2e19f86657a471c5055'
VERSION = 'omni-source-label-pilot-v1:2f1c4d2ddc5b299f0a86161a59635c719b3f5b84646c49d9e78a484f4481573e'
DS = ROOT / 'H3-posttrain/scripts_omnidance/exp/h3ar/ds_zero3_actckpt.json'
HISTORY = MS / 'h3_video_interface_repeatlast124_v2_queue'
HOST = '28.216.19.137'
GPUS = list(range(8))
QUEUE = None
FROZEN = {}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(name, value):
    with (QUEUE / name).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def event(name, **value):
    value = dict(event=name, manager_pid=os.getpid(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **value)
    print(json.dumps(value), flush=True)
    if QUEUE is not None and QUEUE.is_dir():
        with (QUEUE / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value) + '\n')
            stream.flush()
            os.fsync(stream.fileno())


def verify_frozen():
    assert all(sha(path) == expected for path, expected in FROZEN.items()), 'Frozen helper/manager/config changed'
    assert sha(SOURCE) == SOURCE_SHA and sha(METADATA) == METADATA_SHA, 'Frozen source/metadata changed'


def command_for(destination, anchor):
    return ['env', 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7', 'OMP_NUM_THREADS=2',
            'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'NCCL_IB_DISABLE=1',
            'TORCH_NCCL_ASYNC_ERROR_HANDLING=1', 'OD_DS_ACTCKPT=1',
            '/opt/venv_h3/bin/python', '-B', '-m', 'accelerate.commands.launch',
            '--use_deepspeed', '--deepspeed_config_file', str(DS), '--num_processes', '8',
            '--num_machines', '1', '--main_process_port', '29635', '--mixed_precision', 'bf16',
            '--dynamo_backend', 'no', str(SOURCE), '--task', 'sft:train',
            '--weights_root', '/dev/shm/h3/Ref2VA', '--dataset_base_path', str(CACHE),
            '--caption_version', VERSION, '--output_path', str(destination),
            '--data_scope', 'source_label_train', '--max_train_steps', '100',
            '--height', '1280', '--width', '704', '--num_frames', '124',
            '--dataset_num_workers', '0', '--temporal_policy', 'nearest24_repeat_last124',
            '--seed', '42', '--lora_rank', '32', '--lora_alpha', '64', '--learning_rate', '1e-5',
            '--anchor_lambda', str(anchor), '--gradient_accumulation_steps', '1',
            '--num_epochs', '3', '--save_steps', '100', '--remove_prefix_in_ckpt', 'pipe.dit.']


def training_memory_budget_gib():
    shards = list(Path('/dev/shm/h3/Ref2VA/transformer').glob('model*.safetensors'))
    assert len(shards) == 13
    largest = max(path.stat().st_size for path in shards)
    marker = json.loads((CACHE / 'complete.json').read_text())
    cache_bytes = sum(info['bytes'] for name, info in marker['files'].items() if name.endswith('.pth'))
    # CUDA initialization stages one shard per rank. Include full cache reads and working headroom.
    return math.ceil((8 * largest + cache_bytes) / 2**30) + 80


def cpu_gate(helper, runs):
    assert os.geteuid() == 0 and socket.gethostname() == 'TENCENT64.site', 'Not the authorized N5 host/UID'
    assert sha(HELPER) == HELPER_SHA and sha(WRAPPER) == WRAPPER_SHA and sha(METADATA) == METADATA_SHA
    rows = json.loads(METADATA.read_text())
    assert len(rows) == 512 and all(row['split'] == 'train' and row['trained_caption_approved'] is False and
                                  row['caption_version'] == VERSION for row in rows)
    sidecar = json.loads((METADATA.parent / 'source_label_h3_sidecar.json').read_text())
    assert sidecar['complete'] is True and sidecar['semantic_approval'] is False
    assert sidecar['human_review_completed'] is False and sidecar['original_source_window_mapping'] == 'unknown'
    ds = json.loads(DS.read_text())
    assert ds['zero_optimization']['stage'] == 3 and ds['activation_checkpointing'] == {
        'partition_activations': False, 'cpu_checkpointing': False, 'contiguous_memory_optimization': False}
    history = json.loads((HISTORY / 'train_plan.json').read_text())
    audit = json.loads((HISTORY / 'train_audit.json').read_text())
    assert '--use_deepspeed' in history['command'] and 'OD_DS_ACTCKPT=1' in history['command']
    assert history['command'][history['command'].index('--deepspeed_config_file') + 1] == str(DS)
    assert history['command'][history['command'].index('--num_processes') + 1] == '8'
    assert audit['passed'] is True and audit['actual_gpu_updates'] == 2
    assert all(len(row['optimizer_shards']) == len(row['rng_files']) == 8 for row in audit['records'])
    weights = Path('/dev/shm/h3/Ref2VA')
    assert (weights / 'model_index.json').is_file() and Path('/opt/venv_h3/bin/python').is_file()
    shards = list((weights / 'transformer').glob('model*.safetensors'))
    assert len(shards) == 13
    model_bytes = sum(path.stat().st_size for path in shards)
    largest = max(path.stat().st_size for path in shards)
    snapshot = helper.inspect_resources(GPUS, training_memory_budget_gib())
    helper.reserves_for(GPUS)
    assert 8 * largest + 80 * 2**30 <= snapshot['headroom_bytes'], 'CUDA init still needs bounded CPU staging'
    assert model_bytes + largest + 8 * 2**30 < min(int(row[3]) * 2**20 for row in snapshot['snapshot']), 'BF16 init GPU budget'
    sys.path.insert(0, str(SOURCE.parent))
    import h3_video_sft as h3
    parser = h3.video_sft_parser()
    commands = {name: command_for(destination, anchor) for name, anchor, destination in runs}
    for name, anchor, destination in runs:
        command = commands[name]
        args = parser.parse_args(command[command.index(str(SOURCE)) + 1:])
        assert args.task == 'sft:train' and not args.initialize_model_on_cpu and args.max_train_steps == 100
        assert args.lora_rank == 32 and args.lora_alpha == 64 and args.seed == 42 and args.learning_rate == 1e-5
        assert args.anchor_lambda == anchor and args.use_gradient_checkpointing and not args.resume_training_state
        assert (args.height, args.width, args.num_frames) == (1280, 704, 124)
        assert '--multi_gpu' not in command and '--initialize_model_on_cpu' not in command
    from accelerate.commands.launch import launch_command_parser
    for command in commands.values():
        launch = launch_command_parser().parse_args(command[command.index('accelerate.commands.launch') + 1:])
        assert launch.use_deepspeed and launch.num_processes == 8 and launch.mixed_precision == 'bf16'
        assert launch.deepspeed_config_file == str(DS) and not launch.multi_gpu
    for path in [HELPER, WRAPPER, DS, Path(__file__), HISTORY / 'train_plan.json', HISTORY / 'train_audit.json',
                 SOURCE.with_name('omni_dance_sft.py'), SOURCE.with_name('dance_caption_fusion.py'),
                 ROOT / 'DiffSynth-Studio-upstream-20260930/diffsynth/diffusion/runner.py',
                 ROOT / 'DiffSynth-Studio-upstream-20260930/diffsynth/models/minimax_h3_dit.py']:
        FROZEN[str(path)] = sha(path)
    return h3, dict(commands=commands, host=HOST, queue=str(QUEUE), source_sha256=SOURCE_SHA,
                   observed_source_sha256=sha(SOURCE), source_identity_passed=sha(SOURCE) == SOURCE_SHA,
                   metadata_sha256=METADATA_SHA, helper_sha256=FROZEN,
                   caption_version=VERSION, cache=str(CACHE), cache_queue=str(CACHE_QUEUE),
                   cache_rank_count=4, training_rank_count=8, expected_samples=512,
                   train_outputs={name: str(destination) for name, _, destination in runs},
                   training_model_init_device='cuda', mixed_precision='bf16',
                   model_bytes=model_bytes, largest_shard_bytes=largest, snapshot=snapshot,
                   gpu_forward_performed=False, holder_released=False, semantic_approval=False,
                   human_review_completed=False, original_source_window_mapping='unknown',
                   sequence=['pureFM0_100_successful_updates', 'anchor0.1_100_successful_updates'],
                   automatic_full_or_300_or_anchor1=False)


def wait_for_cache():
    while True:
        source_sha = sha(SOURCE)
        finished_path = CACHE_QUEUE / 'finished.json'
        marker_path = CACHE / 'complete.json'
        finished = json.loads(finished_path.read_text()) if finished_path.is_file() else None
        if finished is not None and finished.get('complete') is not True:
            raise RuntimeError('Original N3 cache finished incomplete; preserve partial assets, no retry')
        if source_sha == SOURCE_SHA and finished is not None and marker_path.is_file():
            return finished
        event('waiting_original_N3_cache', cache=str(CACHE), cache_queue=str(CACHE_QUEUE),
              n3_finished_complete=finished.get('complete') if finished is not None else None,
              cache_complete_marker_present=marker_path.is_file(), source_identity_passed=source_sha == SOURCE_SHA,
              observed_source_sha256=source_sha, expected_source_sha256=SOURCE_SHA,
              blocked_reason='source_sha_changed' if source_sha != SOURCE_SHA else 'original_N3_cache_not_complete',
              actual_training_started=False, gpu_holder_released=False)
        time.sleep(30)


def verify_cache(h3, finished):
    verify_frozen()
    marker = h3.verify_complete_marker(CACHE)
    signature = marker['signature']
    assert finished['complete'] is True and finished['exit_code'] == 0 and finished['samples'] == 512
    assert finished['marker_sha256'] == sha(CACHE / 'complete.json')
    assert marker['scope'] == signature['data_scope'] == 'source_label_train' and signature['dataset_size'] == 512
    assert signature['world_size'] == 4, 'Actual four-rank cache producer required'
    assert signature['caption_version'] == VERSION and signature['temporal_policy'] == 'nearest24_repeat_last124'
    assert signature['sources']['metadata_sha256'] == METADATA_SHA
    assert signature['sources']['code'] == {SOURCE.name: SOURCE_SHA, WRAPPER.name: WRAPPER_SHA}
    assert signature['dataset_metadata_path'] == str(METADATA)
    assert len([name for name in marker['files'] if name.endswith('.pth')]) == 512
    provenance = signature['data_provenance']
    assert provenance['semantic_approval'] is False and provenance['human_review_completed'] is False
    assert provenance['original_source_window_mapping'] == 'unknown' and len(provenance['train_ids']) == 512
    for record in provenance['verified_sources'].values():
        path = Path(record['path'])
        assert path.stat().st_size == record['size_bytes'] and sha(path) == record['sha256']
    for name, info in signature['sources']['original_dit'].items():
        path = Path('/dev/shm/h3/Ref2VA') / name
        assert path.stat().st_size == info['bytes'] and sha(path) == info['sha256']
    save('cache_verified.json', dict(complete=True, marker_sha256=sha(CACHE / 'complete.json'),
        samples=512, cache_rank_count=signature['world_size'], training_rank_count=8,
        cacheworld_is_not_trainingworld=True, semantic_approval=False))
    event('original_N3_cache_verified', samples=512, actual_training_started=False)


def verify_training(h3, destination, anchor):
    state = destination / 'training-state-final-100'
    marker = h3.verify_complete_marker(state)
    signature = marker['signature']
    assert marker['scope'] == 'source_label_train' and marker['progress']['optimizer_updates'] == 100
    assert marker['progress']['signature'] == signature
    assert (signature['anchor_lambda'], signature['world_size'], signature['seed'], signature['lora_rank'],
            signature['lora_alpha'], signature['learning_rate']) == (anchor, 8, 42, 32, 64, 1e-5)
    assert signature['sources']['code'] == {SOURCE.name: SOURCE_SHA, WRAPPER.name: WRAPPER_SHA}
    assert signature['sources']['cache_complete_marker'] == sha(CACHE / 'complete.json')
    files = set(marker['files'])
    assert {'custom_checkpoint_0.pkl', 'scheduler.bin'} <= files
    assert {f'random_states_{rank}.pkl' for rank in GPUS} <= files
    optimizers = sorted(name for name in files if name.endswith('_optim_states.pt'))
    models = sorted(name for name in files if name.endswith('_model_states.pt'))
    assert len(optimizers) == len(models) == 8
    for rank in GPUS:
        assert any(f'zero_pp_rank_{rank}_' in name for name in optimizers)
    import torch
    scheduler = torch.load(state / 'scheduler.bin', map_location='cpu', weights_only=False)
    assert isinstance(scheduler, dict) and scheduler['last_epoch'] >= 100
    assert scheduler['base_lrs'] == [1e-5] and scheduler['_last_lr'] == [1e-5]
    for name in models:
        value = torch.load(state / name, map_location='cpu', weights_only=False, mmap=True)
        assert value['global_steps'] == 100 and value['dp_world_size'] == 8
        assert value['module'] and value['param_shapes']
        del value
    for name in optimizers:
        value = torch.load(state / name, map_location='cpu', weights_only=False, mmap=True)
        optimizer = value['optimizer_state_dict']
        assert optimizer['zero_stage'] == 3 and optimizer['partition_count'] == 8
        adam = optimizer['optimizer_state_dict']['state']
        assert adam and all(int(state_value['step']) == 100 and state_value['exp_avg'].numel() and
            state_value['exp_avg_sq'].numel() for state_value in adam.values())
        del value
    loras = sorted(destination.glob('*.safetensors'))
    assert loras and (destination / 'step-100.safetensors').is_file()
    return dict(complete=True, successful_updates=100, state=str(state),
                marker_sha256=sha(state / 'complete.json'), optimizer_shards=optimizers, model_shards=models,
                rng_rank_count=8, lora_files=[str(path) for path in loras], human_quality_verified=False)


def execute_stage(helper, h3, name, anchor, destination, command):
    verify_frozen()
    assert not destination.exists(), 'Refuse fresh training output overwrite'
    snapshot = helper.inspect_resources(GPUS, training_memory_budget_gib())
    reserves = helper.reserves_for(GPUS)
    save(name + '_admission.json', dict(**snapshot, command=command, source_sha256=SOURCE_SHA))
    for reserve in reserves:
        reserve.touch()
    helper.release_holders(snapshot)
    verify_frozen()
    with (QUEUE / (name + '.log')).open('xb') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        save(name + '_live.json', dict(manager_pid=os.getpid(), launcher_pid=child.pid,
             launcher_identity=helper.identity(child.pid), command=command, host=HOST, gpus=GPUS))
        event('actual_h3_' + name + '_started', launcher_pid=child.pid, gpus=GPUS, output=str(destination))
        while True:
            try:
                code = child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                for reserve in reserves:
                    reserve.touch()
    save(name + '_exit.json', dict(exit_code=code))
    assert code == 0, name + ' failed; preserve partial output, stop queue'
    verify_frozen()
    result = verify_training(h3, destination, anchor)
    save(name + '_verified.json', result)
    event('actual_h3_' + name + '_verified', **result)


def main():
    global QUEUE
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--manager-sha', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--prior-purefm', type=Path, required=True)
    args = parser.parse_args()
    assert str(uuid.UUID(args.run_id)) == args.run_id
    assert sha(__file__) == args.manager_sha, 'Manager source identity changed'
    receipt_path = Path('/opt/h3cache/ms_pilot512_local_20261023_v1_verified.json')
    receipt = json.loads(receipt_path.read_text())
    assert receipt['complete'] is True and receipt['samples'] == 512 and receipt['destination'] == str(CACHE)
    assert receipt['marker_sha256'] == sha(CACHE / 'complete.json')
    QUEUE = MS / ('h3_ms_style_train_queue_n5_anchor_v8_' + args.run_id)
    runs = [(name, anchor, MS / ('h3_ms_style_' + name + '_s100_n5_anchor_v8_' + args.run_id))
            for name, anchor in [('anchor01', 0.1)]]
    assert not QUEUE.exists() and all(not destination.exists() for _, _, destination in runs)
    assert sha(HELPER) == HELPER_SHA
    spec = importlib.util.spec_from_file_location('n5_v2_identity_only', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.SOURCE, helper.OUT, helper.QUEUE, helper.GPUS = SOURCE, CACHE, QUEUE, GPUS
    h3, plan = cpu_gate(helper, runs)
    if args.preflight_only:
        event('cpu_parse_and_budget_gate_passed', **plan)
        return
    QUEUE.mkdir(exist_ok=False)
    save('plan.json', plan)
    save('live.json', dict(manager_pid=os.getpid(), manager_identity=helper.identity(os.getpid()),
                          run_id=args.run_id, host=HOST, manager_sha256=sha(__file__)))
    event('cpu_parse_and_budget_gate_passed', source_identity_passed=plan['source_identity_passed'],
          actual_training_started=False, gpu_holder_released=False, queue=str(QUEUE))
    try:
        # This arm starts from the original base, not the completed pureFM weights.
        # The core independently validates the local cache before training.
        prior_state = args.prior_purefm / 'training-state-final-100'
        prior_marker = json.loads((prior_state / 'complete.json').read_text())
        prior_signature = prior_marker['signature']
        assert prior_marker['progress']['optimizer_updates'] == 100
        assert prior_signature['anchor_lambda'] == 0.0 and prior_signature['world_size'] == 8
        assert prior_signature['sources']['code'] == {SOURCE.name: SOURCE_SHA, WRAPPER.name: WRAPPER_SHA}
        import torch
        scheduler = torch.load(prior_state / 'scheduler.bin', map_location='cpu', weights_only=False)
        assert scheduler['last_epoch'] >= 100 and scheduler['base_lrs'] == [1e-5] and scheduler['_last_lr'] == [1e-5]
        lora = args.prior_purefm / 'step-100.safetensors'
        assert lora.is_file()
        prior_summary = dict(actual_training_updates=100, original_training_exit=0,
            corrected_scheduler_storage='Accelerate scheduler.bin; DeepSpeed lr_scheduler=None is valid',
            prior_state=str(prior_state), prior_complete_marker_sha256=sha(prior_state / 'complete.json'),
            lora=str(lora), lora_sha256=sha(lora), scheduler=scheduler,
            full_checkpoint_content_reverified_this_run=False, human_quality_verified=False)
        save('prior_purefm_training_summary.json', prior_summary)
        event('prior_purefm_training_confirmed_no_retraining', actual_training_updates=100)
        local_marker = json.loads((CACHE / 'complete.json').read_text())
        assert local_marker['scope'] == 'source_label_train' and local_marker['signature']['dataset_size'] == 512
        assert local_marker['signature']['sources']['code'] == {SOURCE.name: SOURCE_SHA, WRAPPER.name: WRAPPER_SHA}
        assert local_marker['signature']['sources']['metadata_sha256'] == METADATA_SHA
        assert local_marker['signature']['temporal_policy'] == 'nearest24_repeat_last124'
        save('local_cache_receipt_checked.json', dict(marker_sha256=sha(CACHE / 'complete.json'), samples=512,
             core_validates_actual_cache_before_train=True, previous_full_file_stage_verification=True))
        for name, anchor, destination in runs:
            execute_stage(helper, h3, name, anchor, destination, plan['commands'][name])
        save('finished.json', dict(complete=True, successful_updates_each=100, human_quality_verified=False))
        event('anchor_stage_complete', successful_updates=100, prior_purefm_retrained=False, human_quality_verified=False)
    except Exception as error:
        save('finished.json', dict(complete=False, error=repr(error), retry_started=False))
        event('blocked_queue_stopped', error=repr(error), partial_assets_preserved=True)
        raise


if __name__ == '__main__':
    main()
