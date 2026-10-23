import argparse
import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import datetime

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
MS = ROOT / 'diff/logs_omnidance/exp/MS'
SOURCE = ROOT / 'H3-posttrain/scripts_omnidance/exp/ms/h3_video_sft.py'
CACHE = Path('/opt/h3_ms_style_cache512_v3')
CACHE_QUEUE = MS / 'h3_ms_style_pilot_cache_queue_v2'
QUEUE = MS / 'h3_ms_style_sequence100_queue_v3'
TRAIN_ROOT = Path('/opt/h3_ms_style_sequence100_v3')
UUIDS = ['GPU-ea3fe93b-ee85-4357-3e35-b482e3c3d8c5', 'GPU-7d83b768-a7c8-40a7-3975-f5619833a307', 'GPU-0412c161-6471-21ae-3b86-88b6937d33a4', 'GPU-230b5ddf-40c2-9507-a457-638e8558f866', 'GPU-b18ddd18-cc10-a6b8-fd4e-d0aaa95d6d5a', 'GPU-82b855ef-9624-74b7-99d6-08f427890067', 'GPU-3989031e-587c-3c7b-091d-ad7f748e2d3a', 'GPU-a47141f9-91c2-af1b-35ce-61592196d7b1']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--wait-gpu4-pid', type=int)
    parser.add_argument('--wait-staging-pid', type=int)
    args = parser.parse_args()
    assert os.geteuid() == 0 and MS.is_dir() and TRAIN_ROOT.parent.is_dir()
    assert not QUEUE.exists() and not TRAIN_ROOT.exists()
    spec = importlib.util.spec_from_file_location('cache_admission_tools', MS / 'h3_ms_style_cache_manager_v1.py')
    admission = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(admission)
    assert admission.sha(SOURCE) == args.source_sha
    QUEUE.mkdir(exist_ok=False)
    admission.QUEUE = QUEUE
    admission.save('plan.json', dict(host='28.216.19.213', gpus=list(range(8)), source_sha256=args.source_sha,
        cache=str(CACHE), output_root=str(TRAIN_ROOT),
        groups=[dict(name='pure_fm', anchor_lambda=0.0), dict(name='anchor_0p1', anchor_lambda=0.1), dict(name='anchor_1', anchor_lambda=1.0)],
        requested_updates_each=100, same_original_base_seed_data=True, semantic_approval=False,
        checkpoint_storage='Existing node-local disk; shared queue stores commands and complete-marker evidence.',
        actual_training_started=False))
    cache_managers = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            argv = [value.decode() for value in (path / 'cmdline').read_bytes().split(b'\0') if value]
            if str(MS / 'h3_ms_style_cache_manager_v1.py') in argv:
                assert path.stat().st_uid == os.getuid() and args.source_sha in argv
                cache_managers.append(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    assert len(cache_managers) <= 1, 'Duplicate cache managers must not receive a second training queue'
    if cache_managers:
        descriptor = os.pidfd_open(cache_managers[0])
        try:
            print(json.dumps(dict(event='sequence100_waiting_existing_cache', cache_manager_pid=cache_managers[0])), flush=True)
            select.select([descriptor], [], [])
        finally:
            os.close(descriptor)
    if args.wait_staging_pid is not None:
        process = Path('/proc') / str(args.wait_staging_pid)
        if process.exists():
            argv = [value.decode() for value in (process / 'cmdline').read_bytes().split(b'\0') if value]
            assert process.stat().st_uid == os.getuid() and str(MS / 'stage_h3_ms_style_cache512_local_v3.py') in argv
            descriptor = os.pidfd_open(args.wait_staging_pid)
            try:
                print(json.dumps(dict(event='sequence100_waiting_cache_local_staging', staging_pid=args.wait_staging_pid)), flush=True)
                select.select([descriptor], [], [])
            finally:
                os.close(descriptor)
    staged = json.loads(Path('/opt/h3_ms_style_cache512_v3_staging.json').read_text())
    assert staged['complete'] is True and staged['destination'] == str(CACHE)
    assert staged['source_marker_sha256'] == staged['destination_marker_sha256'] == '6eba363545c7717155a141755d1c47ed03f1659d028d12305f27d8db6f0e30dc'
    finished = json.loads((CACHE_QUEUE / 'finished.json').read_text())
    marker = json.loads((CACHE / 'complete.json').read_text())
    assert finished['complete'] is True and marker['scope'] == 'source_label_train'
    assert marker['signature']['dataset_size'] == 512
    version = marker['signature']['caption_version']
    admission.save('cache_ready.json', dict(complete=True, marker_sha256=admission.sha(CACHE / 'complete.json'),
        source_scope='source_label_train', samples=512, caption_version=version, semantic_approval=False))
    assert admission.sha(SOURCE) == args.source_sha
    if args.wait_gpu4_pid is not None:
        occupied = Path('/proc') / str(args.wait_gpu4_pid)
        if occupied.exists():
            value = admission.identity(args.wait_gpu4_pid)
            assert value['uids'] == [0] * 4 and value['cuda'] == '4'
            assert len(value['argv']) > 2 and value['argv'][:2] == [
                '/opt/venv_music/bin/python', str(ROOT / 'H3-posttrain/scripts_omnidance/exp/ms/music_direct_token_compare.py')]
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
            assert any([x.strip() for x in line.split(',')] == [UUIDS[4], str(args.wait_gpu4_pid)] for line in apps.splitlines())
            descriptor = os.pidfd_open(args.wait_gpu4_pid)
            try:
                assert admission.identity(args.wait_gpu4_pid) == value
                admission.save('existing_gpu4_wait.json', dict(identity=value, uuid=UUIDS[4],
                    policy='Wait for this existing task; never signal or claim ownership'))
                print(json.dumps(dict(event='H3_training_waiting_existing_GPU4_task', pid=args.wait_gpu4_pid)), flush=True)
                select.select([descriptor], [], [])
            finally:
                os.close(descriptor)
    cg = Path('/sys/fs/cgroup/memory')
    memory = {line.split()[0][:-1]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    headroom = int((cg / 'memory.limit_in_bytes').read_text()) - int((cg / 'memory.usage_in_bytes').read_text())
    assert headroom >= 180 * 2**30 and memory['MemAvailable'] >= 180 * 2**20
    own_cgroup = Path('/proc/self/cgroup').read_text()
    gpu_rows = [[value.strip() for value in line.split(',')] for line in subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True).splitlines()]
    assert len(gpu_rows) == 8 and all(int(row[0]) == i and row[1] == UUIDS[i] and int(row[2]) < 2048 and int(row[3]) >= 80000 for i, row in enumerate(gpu_rows))
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            argv = [value.decode() for value in (path / 'cmdline').read_bytes().split(b'\0') if value]
            assert str(SOURCE) not in argv, 'Existing H3 worker cannot duplicate'
        except (FileNotFoundError, ProcessLookupError):
            pass
    holders = []
    for line in subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True).splitlines():
        uuid, pid, used = [value.strip() for value in line.split(',')]
        value = admission.identity(int(pid))
        assert value['uids'] == [0] * 4 and value['argv'] == admission.HOLDER and value['cgroup'] == own_cgroup
        assert value['cuda'] in [str(i) for i in range(8)] and uuid == UUIDS[int(value['cuda'])] and int(used) <= 1100
        holders.append(dict(identity=value, uuid=uuid))
    assert len({holder['uuid'] for holder in holders}) == len(holders)
    reserves = [Path('/tmp/od_reserve_' + str(i)) for i in range(8)]
    for reserve in reserves:
        assert not reserve.exists() or (reserve.is_file() and not reserve.is_symlink() and reserve.stat().st_uid == os.getuid() and reserve.stat().st_size == 0)
    admission.save('admission.json', dict(gpus=gpu_rows, holders=holders, headroom_bytes=headroom,
        memavailable_bytes=memory['MemAvailable'] * 1024))
    for reserve in reserves:
        reserve.touch()
    fds = []
    try:
        for holder in holders:
            value = holder['identity']
            fd = os.pidfd_open(value['pid'])
            fds.append(fd)
            assert admission.identity(value['pid']) == value
            signal.pidfd_send_signal(fd, signal.SIGTERM)
            signal.pidfd_send_signal(fd, signal.SIGCONT)
        for fd in fds:
            assert select.select([fd], [], [], 30)[0], 'Verified holder failed to exit; no escalation'
    finally:
        for fd in fds:
            os.close(fd)
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True).strip()
    TRAIN_ROOT.mkdir(exist_ok=False)
    results = []
    for name, anchor in [('pure_fm', 0.0), ('anchor_0p1', 0.1), ('anchor_1', 1.0)]:
        out = TRAIN_ROOT / name
        assert not out.exists() and admission.sha(SOURCE) == args.source_sha
        command = ['env', 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7', 'OMP_NUM_THREADS=2', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'NCCL_IB_DISABLE=1', 'TORCH_NCCL_ASYNC_ERROR_HANDLING=1', 'OD_DS_ACTCKPT=1', '/opt/venv_h3/bin/python', '-m', 'accelerate.commands.launch', '--use_deepspeed', '--deepspeed_config_file', str(ROOT / 'H3-posttrain/scripts_omnidance/exp/h3ar/ds_zero3_actckpt.json'), '--num_processes', '8', '--num_machines', '1', '--mixed_precision', 'bf16', '--dynamo_backend', 'no', str(SOURCE), '--task', 'sft:train', '--weights_root', '/dev/shm/h3/Ref2VA', '--dataset_base_path', str(CACHE), '--caption_version', version, '--output_path', str(out), '--data_scope', 'source_label_train', '--learning_rate', '1e-5', '--lora_rank', '32', '--lora_alpha', '64', '--anchor_lambda', str(anchor), '--gradient_accumulation_steps', '1', '--num_epochs', '3', '--max_train_steps', '100', '--save_steps', '100', '--initialize_model_on_cpu', '--temporal_policy', 'nearest24_repeat_last124']
        admission.save(name + '_plan.json', dict(command=command, source_sha256=args.source_sha,
            fresh_initialization=True, optimizer_resume=False))
        with (QUEUE / (name + '.log')).open('xb') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            admission.save(name + '_live.json', dict(manager_pid=os.getpid(), training_launcher_pid=child.pid,
                started_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            print(json.dumps(dict(event='actual_H3_100_group_started', group=name, launcher_pid=child.pid)), flush=True)
            while True:
                try:
                    code = child.wait(timeout=300)
                    break
                except subprocess.TimeoutExpired:
                    for reserve in reserves:
                        reserve.touch()
        final_state = out / 'training-state-final-100' / 'complete.json'
        complete = code == 0 and final_state.is_file()
        if complete:
            progress = json.loads(final_state.read_text())['progress']
            assert progress['optimizer_updates'] == 100
        item = dict(group=name, exit_code=code, complete=complete, output=str(out), semantic_approval=False)
        admission.save(name + '_finished.json', item)
        results.append(item)
        if not complete:
            raise SystemExit(code or 1)
    admission.save('finished.json', dict(complete=True, groups=results, actualhuman=False,
        video_quality_verified=False, inference_needed='N4 fixed three cases per group'))
    print(json.dumps(dict(event='three_H3_groups_complete', results=results)), flush=True)


if __name__ == '__main__':
    main()
