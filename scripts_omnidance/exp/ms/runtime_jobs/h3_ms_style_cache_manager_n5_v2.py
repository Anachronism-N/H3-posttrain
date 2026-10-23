import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
MS = ROOT / 'diff/logs_omnidance/exp/MS'
SOURCE = ROOT / 'H3-posttrain/scripts_omnidance/exp/ms/h3_video_sft.py'
WRAPPER = SOURCE.with_name('h3_video_only.py')
METADATA = MS / 'h3_ms_style_pilot_metadata_v1/h3_video_train.json'
OUT = MS / 'h3_ms_style_pilot_cache512_n5_v2'
QUEUE = MS / 'h3_ms_style_pilot_cache_queue_n5_v2'
GPUS = [0, 1, 2, 3]
UUIDS = ['GPU-6309cd58-c0c9-0d02-b2fb-c5dc0ccb8c3b', 'GPU-22fb8f46-d96c-75d5-14c2-358f85137e25', 'GPU-f4e31ddd-8e78-0c88-7a78-382f05db93f7', 'GPU-d291bcac-a145-8bdd-192f-770181cf1969', 'GPU-177f51a0-3db5-c8a3-f3fd-534fc2555198', 'GPU-0960b0b3-b6af-886e-455d-609c29bf2b03', 'GPU-0294b607-4f86-fe41-6c19-1242f5cdb1ed', 'GPU-ec445853-b769-db7d-888b-58c4735153ad']
HOLDER = ['/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/envs/longlive/bin/python', str(ROOT / 'diff/scripts_omnidance/exp/hold_gpu.py')]


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


def identity(pid):
    root = Path('/proc') / str(pid)
    environment = dict(value.split('=', 1) for value in (root / 'environ').read_bytes().decode().split('\0') if '=' in value)
    return dict(pid=pid, uids=[int(value) for line in (root / 'status').read_text().splitlines() if line.startswith('Uid:') for value in line.split()[1:]],
        argv=[value.decode() for value in (root / 'cmdline').read_bytes().split(b'\0') if value],
        cuda=environment.get('CUDA_VISIBLE_DEVICES', environment.get('OCCUPY_GPU')),
        cgroup=(root / 'cgroup').read_text(), start_ticks=(root / 'stat').read_text().rsplit(')', 1)[1].split()[19])


HOST = '28.216.19.137'
TRAIN_RUNS = [('purefm', 0.0, MS / 'h3_ms_style_pilot_purefm_s100_n5_v2'),
              ('anchor01', 0.1, MS / 'h3_ms_style_pilot_anchor01_s100_n5_v2')]


def inspect_resources(gpus, minimum_gib):
    own_cgroup = Path('/proc/self/cgroup').read_text()
    memory = {line.split()[0][:-1]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    cg = Path('/sys/fs/cgroup/memory')
    limit, usage = [int((cg / name).read_text()) for name in ('memory.limit_in_bytes', 'memory.usage_in_bytes')]
    assert limit - usage >= minimum_gib * 2**30 and memory['MemAvailable'] >= minimum_gib * 2**20, 'Insufficient host/cgroup memory'
    all_gpus = [[value.strip() for value in line.split(',')] for line in subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True).splitlines()]
    selected = {int(row[0]): row for row in all_gpus if int(row[0]) in gpus}
    assert set(selected) == set(gpus)
    for gpu in gpus:
        row = selected[gpu]
        assert row[1] == UUIDS[gpu] and int(row[2]) < 2048 and int(row[3]) >= 80000, 'Unexpected GPU identity/occupancy'
    for root in Path('/proc').iterdir():
        if not root.name.isdigit():
            continue
        try:
            argv = [value.decode() for value in (root / 'cmdline').read_bytes().split(b'\0') if value]
            assert str(SOURCE) not in argv and str(WRAPPER) not in argv, 'Existing H3 task cannot duplicate'
        except (FileNotFoundError, ProcessLookupError):
            continue
    holders = []
    for line in subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True).splitlines():
        uuid, pid, used = [value.strip() for value in line.split(',')]
        if uuid not in [UUIDS[gpu] for gpu in gpus]:
            continue
        value = identity(int(pid))
        assert value['uids'] == [0] * 4 and value['argv'] == HOLDER and value['cgroup'] == own_cgroup, 'Unknown GPU process; no signal permitted'
        assert value['cuda'] in [str(gpu) for gpu in gpus] and UUIDS[int(value['cuda'])] == uuid and int(used) <= 1100
        holders.append(dict(identity=value, uuid=uuid, used_MiB=int(used)))
    assert len({holder['uuid'] for holder in holders}) == len(holders)
    return dict(host=HOST, hostname=subprocess.check_output(['hostname'], text=True).strip(),
                gpus=gpus, snapshot=[selected[gpu] for gpu in gpus], holders=holders, cgroup=own_cgroup,
                cgroup_limit_bytes=limit, cgroup_usage_bytes=usage, headroom_bytes=limit-usage,
                memavailable_bytes=memory['MemAvailable'] * 1024, minimum_headroom_GiB=minimum_gib)


def release_holders(snapshot):
    fds = []
    try:
        for holder in snapshot['holders']:
            value = holder['identity']
            fd = os.pidfd_open(value['pid'])
            fds.append(fd)
            assert identity(value['pid']) == value
            signal.pidfd_send_signal(fd, signal.SIGTERM)
            signal.pidfd_send_signal(fd, signal.SIGCONT)
        for fd in fds:
            assert select.select([fd], [], [], 30)[0], 'Exact holder did not exit; no escalation'
    finally:
        for fd in fds:
            os.close(fd)
    for line in subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True).splitlines():
        assert line.split(',')[0].strip() not in [UUIDS[gpu] for gpu in snapshot['gpus']]


def reserves_for(gpus):
    reserves = [Path('/tmp/od_reserve_' + str(gpu)) for gpu in gpus]
    for reserve in reserves:
        assert not reserve.exists() or (reserve.is_file() and not reserve.is_symlink() and reserve.stat().st_uid == os.getuid() and reserve.stat().st_size == 0)
    return reserves


def command_for(gpus, version, task, destination, anchor=0.1):
    command = ['env', 'CUDA_VISIBLE_DEVICES=' + ','.join(map(str, gpus)), 'OMP_NUM_THREADS=2',
               'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'NCCL_IB_DISABLE=1',
               'TORCH_NCCL_ASYNC_ERROR_HANDLING=1', '/opt/venv_h3/bin/python', '-B', '-m',
               'accelerate.commands.launch', '--multi_gpu', '--num_processes', str(len(gpus)),
               '--num_machines', '1', '--main_process_port', '29635', '--mixed_precision', 'bf16',
               '--dynamo_backend', 'no', str(SOURCE), '--task', task, '--weights_root', '/dev/shm/h3/Ref2VA',
               '--dataset_base_path', str(ROOT if task == 'sft:data_process' else OUT),
               '--caption_version', version, '--output_path', str(destination), '--data_scope', 'source_label_train',
               '--max_train_steps', '100', '--height', '1280', '--width', '704', '--num_frames', '124',
               '--dataset_num_workers', '0', '--temporal_policy', 'nearest24_repeat_last124',
               '--seed', '42', '--lora_rank', '32', '--lora_alpha', '64', '--learning_rate', '1e-5',
               '--anchor_lambda', str(anchor), '--gradient_accumulation_steps', '1', '--save_steps', '100',
               '--remove_prefix_in_ckpt', 'pipe.dit.']
    if task == 'sft:data_process':
        command += ['--dataset_metadata_path', str(METADATA), '--initialize_model_on_cpu']
    return command


def execute_stage(stage, command, gpus, minimum_gib, args):
    assert sha(SOURCE) == args.source_sha and sha(METADATA) == args.metadata_sha
    snapshot = inspect_resources(gpus, minimum_gib)
    reserves = reserves_for(gpus)
    save(stage + '_admission.json', dict(**snapshot, command=command))
    for reserve in reserves:
        reserve.touch()
    release_holders(snapshot)
    assert sha(SOURCE) == args.source_sha and sha(METADATA) == args.metadata_sha
    with (QUEUE / (stage + '.log')).open('xb') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        live = dict(manager_pid=os.getpid(), launcher_pid=child.pid, host=HOST, gpus=gpus,
                    command=command, started_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(stage + '_live.json', live)
        print(json.dumps(dict(event='actual_h3_' + stage + '_started', **live)), flush=True)
        while True:
            try:
                code = child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                for reserve in reserves:
                    reserve.touch()
    save(stage + '_exit.json', dict(exit_code=code, source_sha256=sha(SOURCE)))
    assert sha(SOURCE) == args.source_sha and sha(METADATA) == args.metadata_sha
    assert code == 0, stage + ' failed; preserve partial output and stop queue'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--metadata-sha', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    assert os.geteuid() == 0 and MS.is_dir() and not OUT.exists() and not QUEUE.exists()
    assert all(not destination.exists() for _, _, destination in TRAIN_RUNS)
    assert sha(SOURCE) == args.source_sha and sha(METADATA) == args.metadata_sha
    assert sha(WRAPPER) == '89439d733d62f522bb3dd993bc456541b3ca15ead34ff2e19f86657a471c5055'
    rows = json.loads(METADATA.read_text())
    assert len(rows) == 512 and all(row['split'] == 'train' and row['trained_caption_approved'] is False for row in rows)
    versions = {row['caption_version'] for row in rows}
    assert len(versions) == 1
    version = next(iter(versions))
    admission = json.loads((METADATA.parent / 'data_admission.json').read_text())
    assert admission['complete'] is True and admission['semantic_approval'] is False
    assert admission['frozen_sources']['h3_video_sft.py']['sha256'] == args.source_sha
    assert admission['data_provenance']['metadata']['sha256'] == args.metadata_sha
    weights = Path('/dev/shm/h3/Ref2VA')
    assert (weights / 'model_index.json').is_file() and Path('/opt/venv_h3/bin/python').is_file()
    model_bytes = {}
    for name, expected in [('text_encoder', 14), ('video_vae/source', 1), ('transformer', 13)]:
        files = list((weights / name).glob('model*.safetensors'))
        assert len(files) == expected
        model_bytes[name] = sum(path.stat().st_size for path in files)
    cache_source_weight_bytes = model_bytes['text_encoder'] + model_bytes['video_vae/source']
    assert 4 * cache_source_weight_bytes + 80 * 2**30 <= 380 * 2**30
    snapshot = inspect_resources(GPUS, 380)
    reserves_for(list(range(8)))
    commands = dict(cache=command_for(GPUS, version, 'sft:data_process', OUT),
                    **{name: command_for(list(range(8)), version, 'sft:train', destination, anchor)
                       for name, anchor, destination in TRAIN_RUNS})
    # Parse the real training CLI only; do not repeat synthetic/GPU smoke or media audits.
    import sys
    sys.path.insert(0, str(SOURCE.parent))
    import h3_video_sft as h3
    real_parser = h3.video_sft_parser()
    for name, command in commands.items():
        config = real_parser.parse_args(command[command.index(str(SOURCE)) + 1:])
        assert config.max_train_steps == 100 and config.seed == 42 and config.lora_rank == 32 and config.lora_alpha == 64
        assert config.learning_rate == 1e-5 and config.data_scope == 'source_label_train'
        assert config.height == 1280 and config.width == 704 and config.num_frames == 124
        assert config.anchor_lambda == (0.0 if name == 'purefm' else 0.1)
    plan = dict(**snapshot, commands=commands, source_sha256=args.source_sha, metadata_sha256=args.metadata_sha,
                manager_sha256=sha(__file__), caption_version=version, expected_samples=512,
                cache_output=str(OUT), train_outputs={name: str(destination) for name, _, destination in TRAIN_RUNS},
                counts=dict(train=512, val=128), data_scope='source_label_train', semantic_approval=False,
                human_review_completed=False, original_source_window_mapping='unknown', model_bytes=model_bytes,
                cache_rank_count=4, training_rank_count=8, training_model_init_device='cuda',
                cache_source_weight_bytes_per_rank=cache_source_weight_bytes,
                same_initialization_protocol='same original BF16 DiT content/source, LoRA rank/alpha, seed42; no previous-trained LoRA/resume',
                sequence=['cache512', 'pureFM100', 'anchor0.1_100'], automatic_full_or_300_or_anchor1=False)
    if args.preflight_only:
        print(json.dumps(dict(event='cpu_plan_budget_gate_passed', gpu_forward_performed=False,
                              holder_released=False, actual_training_started=False, **plan)), flush=True)
        return
    QUEUE.mkdir(exist_ok=False)
    save('plan.json', plan)
    execute_stage('cache', commands['cache'], GPUS, 380, args)
    marker = h3.verify_complete_marker(OUT)
    assert marker['scope'] == 'source_label_train' and marker['signature']['dataset_size'] == 512
    assert marker['signature']['sources']['metadata_sha256'] == args.metadata_sha
    assert marker['signature']['sources']['code'][SOURCE.name] == args.source_sha
    assert marker['signature']['temporal_policy'] == 'nearest24_repeat_last124'
    assert len([path for path in OUT.rglob('*.pth')]) == 512
    save('cache_verified.json', dict(complete=True, samples=512, marker_sha256=sha(OUT / 'complete.json'),
                                    semantic_approval=False, human_review_completed=False))
    for name, anchor, destination in TRAIN_RUNS:
        execute_stage(name, commands[name], list(range(8)), 380, args)
        state = destination / 'training-state-final-100'
        marker = h3.verify_complete_marker(state)
        assert marker['progress']['optimizer_updates'] == 100
        assert marker['signature']['anchor_lambda'] == anchor and marker['signature']['world_size'] == 8
        assert marker['signature']['seed'] == 42 and marker['signature']['lora_rank'] == 32 and marker['signature']['lora_alpha'] == 64
        assert marker['signature']['sources']['code'][SOURCE.name] == args.source_sha
        save(name + '_verified.json', dict(complete=True, successful_updates=100,
             full_Adam_RNG_state=str(state), marker_sha256=sha(state / 'complete.json'),
             lora_files=[str(path) for path in destination.glob('*.safetensors')], human_quality_verified=False))
        print(json.dumps(dict(event='actual_h3_' + name + '_verified', optimizer_updates=100, output=str(destination))), flush=True)
    save('finished.json', dict(complete=True, successful_updates_each=100, human_quality_verified=False))


if __name__ == '__main__':
    main()
