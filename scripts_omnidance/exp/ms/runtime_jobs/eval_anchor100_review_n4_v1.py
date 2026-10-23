"""Wait for independently verified anchor100, then infer fixed GT3 on N4 GPU2."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time
import uuid

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
MS = ROOT / 'diff/logs_omnidance/exp/MS'
QUEUE = MS / 'h3_ms_style_train_queue_n5_anchor_v8_6452f0c2-0fbe-4f6e-b79a-01a169818bf5'
TRAIN = MS / 'h3_ms_style_anchor01_s100_n5_anchor_v8_6452f0c2-0fbe-4f6e-b79a-01a169818bf5'
OUT = MS / 'review_h3_anchor01_s100_n4_v1'
GT = MS / 'omni_h3_ms_style_eval3_v1/h3_gt_caption3.json'
CACHE = MS / 'h3_ms_style_eval3_gt_conditions_v1'
INFER = ROOT / 'H3-posttrain/scripts_omnidance/exp/ms/h3_video_infer_chunked_lora_v1.py'
CORE = INFER.with_name('h3_video_infer.py')
INFER_SHA = 'c611d16c26c38e918d4c91560f7fa72e6e88ad93b43ecad1b48fd80a91deb665'
CORE_SHA = '77bafd9f6165f80328d4e272c901b21fbf45da3898161bcc039a572e1c86a2f8'
SOURCE_SHA = '247e77eb365685a92680bd3934ac172c7d81242f06fb28f7f4aa665130233c63'
WRAPPER_SHA = '89439d733d62f522bb3dd993bc456541b3ca15ead34ff2e19f86657a471c5055'
GT_SHA = '17ff545eff693e19fd18fb60f10f6ae452636e14a0f06a0b6a138e0cdbad8020'
HOST = '28.216.19.143'
GPU = 2
GPU_UUID = 'GPU-f9c8016f-56cb-83c2-7230-f3d4868e9264'
IDS = ['wJPptuhclZmKAEWs', 'wJPrpFsThWAcBQIw', 'wJQTclrlOLEYYfTB']
HOLDER = ['/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/envs/longlive/bin/python', str(ROOT / 'diff/scripts_omnidance/exp/hold_gpu.py')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def event(name, **value):
    value.update(event=name, manager_pid=os.getpid(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    print(json.dumps(value), flush=True)
    with (OUT / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(value) + '\n')
        stream.flush()


def identity(pid):
    p = Path('/proc') / str(pid)
    env = dict(v.split('=', 1) for v in (p / 'environ').read_bytes().decode().split(chr(0)) if '=' in v)
    return dict(pid=pid,
        uids=[int(v) for line in (p / 'status').read_text().splitlines() if line.startswith('Uid:') for v in line.split()[1:]],
        argv=[v.decode() for v in (p / 'cmdline').read_bytes().split(bytes([0])) if v],
        start_ticks=(p / 'stat').read_text().rsplit(')', 1)[1].split()[19],
        cuda=env.get('CUDA_VISIBLE_DEVICES', env.get('OCCUPY_GPU')),
        cgroup=(p / 'cgroup').read_text())


def verified_training():
    finished = QUEUE / 'finished.json'
    verified = QUEUE / 'anchor01_verified.json'
    if not finished.is_file():
        return None
    terminal = json.loads(finished.read_text())
    if terminal.get('complete') is not True:
        raise RuntimeError('Anchor training or its full checkpoint verification failed; preserve output, do not infer')
    assert verified.is_file()
    receipt = json.loads(verified.read_text())
    assert receipt['complete'] is True and receipt['successful_updates'] == 100
    assert receipt['rng_rank_count'] == 8 and len(receipt['optimizer_shards']) == len(receipt['model_shards']) == 8
    state = TRAIN / 'training-state-final-100'
    assert receipt['state'] == str(state) and receipt['marker_sha256'] == sha(state / 'complete.json')
    marker = json.loads((state / 'complete.json').read_text())
    sig = marker['signature']
    assert marker['scope'] == 'source_label_train' and marker['progress']['optimizer_updates'] == 100
    assert marker['progress']['signature'] == sig
    assert (sig['anchor_lambda'], sig['world_size'], sig['seed'], sig['lora_rank'], sig['lora_alpha'], sig['learning_rate']) == (0.1, 8, 42, 32, 64, 1e-5)
    assert sig['sources']['code'] == {'h3_video_sft.py': SOURCE_SHA, 'h3_video_only.py': WRAPPER_SHA}
    files = marker['files']
    assert {'scheduler.bin', 'custom_checkpoint_0.pkl'} <= set(files)
    assert {f'random_states_{rank}.pkl' for rank in range(8)} <= set(files)
    assert set(receipt['optimizer_shards'] + receipt['model_shards']) <= set(files)
    for name, record in files.items():
        assert (state / name).is_file() and (state / name).stat().st_size == record['bytes']
    lora = TRAIN / 'step-100.safetensors'
    assert str(lora) in receipt['lora_files'] and lora.is_file()
    return dict(receipt=receipt, receipt_sha256=sha(verified), state_marker_sha256=sha(state / 'complete.json'), lora=lora)


def frozen(manager_sha):
    assert sha(__file__) == manager_sha and sha(INFER) == INFER_SHA and sha(CORE) == CORE_SHA
    assert sha(GT) == GT_SHA
    assert sha(CORE.with_name('h3_video_sft.py')) == SOURCE_SHA
    assert sha(CORE.with_name('h3_video_only.py')) == WRAPPER_SHA


def admit():
    cg = Path('/sys/fs/cgroup/memory')
    room = int((cg / 'memory.limit_in_bytes').read_text()) - int((cg / 'memory.usage_in_bytes').read_text())
    mem = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))) * 1024
    if min(room, mem) < 180 * 2**30:
        event('waiting_cpu_headroom', headroom_bytes=room, available_bytes=mem, required_bytes=180*2**30)
        return None
    pipelines = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            v = identity(int(p.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        if any(Path(a).name in {'h3_video_infer.py', 'h3_video_infer_chunked_lora_v1.py'} for a in v['argv']):
            assert v['uids'] == [0]*4 and v['cuda'] is not None
            assert str(GPU) not in v['cuda'].split(','), 'Target GPU already runs inference; no duplicate'
            pipelines.append(v)
    if len(pipelines) >= 2:
        event('waiting_two_pipeline_node_limit', active_pipelines=pipelines)
        return None
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()
    gpu = [v.strip() for line in rows if line.split(',')[0].strip() == str(GPU) for v in line.split(',')]
    assert gpu[0] == str(GPU) and gpu[1] == GPU_UUID
    reserve = Path(f'/tmp/od_reserve_{GPU}')
    assert not reserve.exists() or (reserve.is_file() and not reserve.is_symlink() and reserve.stat().st_uid == os.getuid() and reserve.stat().st_size == 0)
    holders = []
    for line in subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True).splitlines():
        gpu_uuid, pid, used = [v.strip() for v in line.split(',')]
        if gpu_uuid != GPU_UUID:
            continue
        v = identity(int(pid))
        assert v['uids'] == [0]*4 and v['argv'] == HOLDER and v['cuda'] == str(GPU)
        assert v['cgroup'] == Path('/proc/self/cgroup').read_text() and float(used) <= 1100
        holders.append(v)
    assert len(holders) <= 1 and float(gpu[2]) <= 1200 and float(gpu[3]) >= 78000
    save(OUT / ('admission_' + str(time.time_ns()) + '.json'), dict(host=HOST, gpu=GPU, uuid=GPU_UUID, holders=holders, headroom_bytes=room, existing_pipelines=pipelines))
    reserve.touch()
    for v in holders:
        fd = os.pidfd_open(v['pid'])
        try:
            assert identity(v['pid']) == v
            signal.pidfd_send_signal(fd, signal.SIGTERM)
            assert select.select([fd], [], [], 30)[0]
        finally:
            os.close(fd)
    return reserve


def main(args):
    assert str(uuid.UUID(args.run_id)) == args.run_id and os.getuid() == 0
    frozen(args.manager_sha)
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True).splitlines()
    assert [v.strip() for line in gpu_rows if line.split(',')[0].strip() == str(GPU) for v in line.split(',')] == [str(GPU), GPU_UUID], 'Not the authorized N4 GPU2 identity'
    assert not OUT.exists(), 'Refuse overwrite or duplicate evaluation'
    assert [row['id'] for row in json.loads(GT.read_text())['records']] == IDS
    assert json.loads((CACHE / 'run_complete.json').read_text())['completed_ids'] == IDS
    OUT.mkdir()
    save(OUT / 'manager_live.json', dict(identity=identity(os.getpid()), run_id=args.run_id, source_sha256=args.manager_sha, host=HOST, gpu=GPU, uuid=GPU_UUID))
    try:
        result = None
        for attempt in range(2880):
            frozen(args.manager_sha)
            result = verified_training()
            if result is not None:
                break
            if attempt % 10 == 0:
                event('waiting_actual100_and_full_adam_rng_lora_verification', queue=str(QUEUE), no_inference_started=True)
            time.sleep(30)
        assert result is not None, 'Training verification wait exhausted; no unverified inference'
        save(OUT / 'training_verified_receipt.json', {**result, 'lora': str(result['lora']), 'full_checkpoint_reverified_by_training_manager': True, 'adam_loaded_on_eval_node': False})
        lora = Path('/opt/h3_review_anchor01_step100_' + args.run_id + '.safetensors')
        assert not lora.exists()
        remote_hash = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '28.216.19.137', 'sha256sum', str(result['lora'])], text=True).split()[0]
        assert len(remote_hash) == 64 and all(c in '0123456789abcdef' for c in remote_hash)
        subprocess.run(['scp', '-O', '-p', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '28.216.19.137:' + str(result['lora']), str(lora)], check=True)
        assert sha(lora) == remote_hash
        from safetensors import safe_open
        with safe_open(str(lora), framework='pt', device='cpu') as stream:
            keys = list(stream.keys())
            assert keys and all('.lora_' in key for key in keys)
            assert all((key.replace('.lora_A.', '.lora_B.') if '.lora_A.' in key else key.replace('.lora_B.', '.lora_A.')) in keys for key in keys)
        save(OUT / 'weight_verified.json', dict(lora=str(lora), original_lora=str(result['lora']), sha256=remote_hash, actual_training_updates=100, tensor_keys=len(keys)))
        reserve = None
        for attempt in range(2880):
            frozen(args.manager_sha)
            reserve = admit()
            if reserve is not None:
                break
            time.sleep(30)
        assert reserve is not None, 'Resource admission wait exhausted'
        argv = ['/opt/venv_h3/bin/python', '-B', '-u', str(INFER), '--caption-json', str(GT), '--weights-root', '/dev/shm/h3/Ref2VA', '--limit', '3', '--steps', '50', '--cfg', '1', '--seed', '42', '--out', str(OUT / 'videos'), '--condition-cache', str(CACHE), '--lora', str(lora)]
        save(OUT / 'plan.json', dict(command=argv, ids=IDS, gpu=GPU, uuid=GPU_UUID, cache=str(CACHE), cache_reused=True, internal_audio=False, human_quality_accepted=False))
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(GPU), OMP_NUM_THREADS='2', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', DIFFSYNTH_SKIP_DOWNLOAD='True')
        with (OUT / 'video.log').open('x') as stream:
            frozen(args.manager_sha)
            child = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT)
            save(OUT / 'video_live.json', dict(manager_pid=os.getpid(), child_pid=child.pid, identity=identity(child.pid), command=argv, host=HOST, gpu=GPU, uuid=GPU_UUID))
            event('actual_anchor100_video_started', child_pid=child.pid, gpu=GPU, actual_training_updates=100)
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    reserve.touch()
        save(OUT / 'video_exit.json', dict(exit_code=code))
        assert code == 0, 'Video stage failed; preserve all partial output'
        frozen(args.manager_sha)
        report = json.loads((OUT / 'videos/run_complete.json').read_text())
        assert report['complete'] and report['completed_ids'] == IDS
        for sid in IDS:
            audit = json.loads((OUT / 'videos' / (sid + '.audit.json')).read_text())
            assert audit['complete'] and audit['media']['complete_decode'] and audit['media']['frames'] == 124
            assert audit['media']['width'] == 704 and audit['media']['height'] == 1280 and audit['media']['audio_streams'] == 0
            assert not any(audit['audio_guard']['calls'].values()) and audit['sampling']['video_scheduler_updates'] == 50
            assert audit['runtime_lora_implementation']['source_sha256'] == INFER_SHA
        save(OUT / 'finished.json', dict(complete=True, ids=IDS, actual_training_updates=100, human_quality_accepted=False))
        event('anchor100_all_three_videos_verified', ids=IDS, human_quality_accepted=False)
    except Exception as error:
        save(OUT / 'finished.json', dict(complete=False, error=repr(error), partial_outputs_preserved=True, retry_started=False))
        event('anchor_eval_blocked_preserve_output', error=repr(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--manager-sha', required=True)
    main(parser.parse_args())
