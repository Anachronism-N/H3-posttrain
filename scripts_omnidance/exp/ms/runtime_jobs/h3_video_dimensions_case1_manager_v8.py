import datetime
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import traceback

MS = Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS')
SCRIPT = Path('/apdcephfs_gy4/share_302533218/cedricnie/H3-posttrain/scripts_omnidance/exp/ms/h3_video_infer.py')
PYTHON = '/opt/venv_h3/bin/python'
SID = 'FqlHJVDlFGLKCUJS'
BASE = MS / 'h3_video_baseline_fused_case1_humanrequest_v7.json'
BASE_SHA = 'b367f18b1406092f6328f3b2c0e9d294b9e3d8eaa27f0f66e2373f394bebd143'
SCRIPT_SHA = '77bafd9f6165f80328d4e272c901b21fbf45da3898161bcc039a572e1c86a2f8'
WRAPPER_SHA = '89439d733d62f522bb3dd993bc456541b3ca15ead34ff2e19f86657a471c5055'
UUIDS = {0: 'GPU-93dc1091-2ba5-13bf-9c46-e716abe867cc', 1: 'GPU-f940954a-c4f3-f1b4-25fe-9c2ad89a9e93'}
HOLDER_ARGV = ['/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/envs/longlive/bin/python', '/apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/hold_gpu.py']
QUEUE = MS / 'h3_video_dimensions_case1_queue_v8'
CONFIGS = [
    dict(gpu=0, name='focus_mood', version='rich-ms-dimensions-focus-mood-v8', append='The movement is led by the arms, with graceful, dreamy expression.'),
    dict(gpu=1, name='level_travel', version='rich-ms-dimensions-level-travel-v8', append='The dancer stays upright and mostly in place, without jumps.'),
]
for config in CONFIGS:
    stem = 'h3_video_dimensions_' + config['name'] + '_case1'
    config.update(caption=str(MS / (stem + '_v8.json')), cache=str(MS / (stem + '_cache_v8')), video=str(MS / (stem + '_video_v8')))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def save(name, value):
    with (QUEUE / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def event(kind, **values):
    value = dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), event=kind, manager_pid=os.getpid(), **values)
    text = json.dumps(value, ensure_ascii=False)
    print(text, flush=True)
    with (QUEUE / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(text + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return value


def process_identity(pid):
    root = Path('/proc') / str(pid)
    status = (root / 'status').read_text()
    uid = [int(value) for line in status.splitlines() if line.startswith('Uid:') for value in line.split()[1:]]
    raw_env = (root / 'environ').read_bytes()
    environment = dict(item.decode().split('=', 1) for item in raw_env.split(b'\0') if item)
    stat = (root / 'stat').read_text()
    return dict(pid=pid, uids=uid, argv=[item.decode() for item in (root / 'cmdline').read_bytes().split(b'\0') if item], executable=str((root / 'exe').resolve()), cwd=str((root / 'cwd').resolve()), start_ticks=int(stat[stat.rfind(')') + 2:].split()[19]), env_sha256=hashlib.sha256(raw_env).hexdigest(), selected_env={key: environment.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')})


def snapshot():
    gpu_text = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
    process_text = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
    gpus = {}
    for line in gpu_text.splitlines():
        index, uuid, total, used = [part.strip() for part in line.split(',')]
        if int(index) in UUIDS:
            if uuid != UUIDS[int(index)]:
                raise RuntimeError('GPU UUID changed: ' + line)
            gpus[int(index)] = dict(index=int(index), uuid=uuid, total_mib=int(total), used_mib=int(used), processes=[])
    if set(gpus) != set(UUIDS):
        raise RuntimeError('Target GPUs not all present')
    for line in process_text.splitlines():
        uuid, pid, used = [part.strip() for part in line.split(',')]
        for gpu in gpus.values():
            if gpu['uuid'] == uuid:
                gpu['processes'].append(dict(gpu_uuid=uuid, pid=int(pid), used_mib=int(used)))
    return gpus


def release_verified_holder(gpu, process):
    pid = process['pid']
    descriptor = os.pidfd_open(pid)
    try:
        identity = process_identity(pid)
        if identity['uids'] != [0, 0, 0, 0] or identity['argv'] != HOLDER_ARGV or identity['selected_env']['CUDA_VISIBLE_DEVICES'] != str(gpu) or process['used_mib'] > 1100:
            raise RuntimeError('Unknown target GPU process; refusing release: ' + json.dumps(identity))
        fresh = snapshot()[gpu]
        if fresh['processes'] != [process] or process_identity(pid) != identity:
            raise RuntimeError('Holder changed during identity check; refusing release')
        event('holder_release_authorized', gpu=gpu, uuid=UUIDS[gpu], process=process, identity=identity)
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        if not select.select([descriptor], [], [], 10)[0]:
            raise RuntimeError('Exact holder did not exit after SIGTERM; no escalation')
        event('holder_exited', gpu=gpu, pid=pid)
    finally:
        os.close(descriptor)


def admit(stage):
    current = snapshot()
    event('fresh_gpu_snapshot', stage=stage, gpus=current)
    for gpu in sorted(UUIDS):
        processes = current[gpu]['processes']
        if len(processes) > 1:
            raise RuntimeError('Multiple/unknown target GPU processes; refusing admission')
        if processes:
            release_verified_holder(gpu, processes[0])
    fresh = snapshot()
    for gpu in sorted(UUIDS):
        entry = fresh[gpu]
        if entry['processes'] or entry['used_mib'] > 1024 or entry['total_mib'] < 80000:
            event('admission_blocked', stage=stage, gpu=gpu, snapshot=entry)
            raise RuntimeError('Target GPU busy after exact release; refusing launch')
    event('gpu_admission_passed', stage=stage, gpus=fresh)


def command(config, stage):
    values = ['env', 'CUDA_VISIBLE_DEVICES=' + str(config['gpu']), 'OMP_NUM_THREADS=2', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', PYTHON, '-u', str(SCRIPT), '--caption-json', config['caption'], '--weights-root', '/dev/shm/h3/Ref2VA', '--out', config[stage]]
    if stage == 'cache':
        values.append('--cache-only')
    else:
        values.extend(['--condition-cache', config['cache']])
    return values


def validate_stage(config, stage):
    out = Path(config[stage])
    run, case = read(out / 'run_complete.json'), read(out / (SID + '.audit.json'))
    row = read(config['caption'])
    if run['complete'] is not True or run['completed_ids'] != [SID] or case['complete'] is not True or case['id'] != SID or case['caption_version'] != config['version'] or case['motion_caption'] != row['motion_caption']:
        raise RuntimeError('Incomplete/mismatched run or case audit')
    if run['geometry'] != dict(height=1280, width=704, num_frames=124, fps=24) or (run['steps'], run['seed'], run['cfg'], run['flow_shift']) != (50, 42, 1, 12):
        raise RuntimeError('Sampling protocol changed')
    if run['source_code']['sha256'] != SCRIPT_SHA or run['wrapper_source']['sha256'] != WRAPPER_SHA or run['caption_source']['sha256'] != sha(config['caption']) or run['old_y_reused'] or run['lora_source'] is not None:
        raise RuntimeError('Code/condition provenance changed')
    if case['sources']['image']['sha256'] != 'd9d8d4f0ccf59d47ab7feb20014fe4d17e68fffeaf99fdce9bd44d90a17b4ef1' or case['sources']['image']['sha256'] != sha(row['image_path']):
        raise RuntimeError('Reference differs from base v7')
    if any(case['audio_guard']['calls'].values()) or not case['audio_guard']['audio_vae_is_none'] or not case['audio_guard']['scheduler_audio_is_none'] or case['packed']['audio_pos_numel'] or case['packed']['tag2_count']:
        raise RuntimeError('Audio exclusion audit failed')
    if sha(case['condition_cache']['path']) != case['condition_cache']['sha256']:
        raise RuntimeError('Condition cache SHA differs')
    observed = case['audio_guard']['observed_video_calls']
    if stage == 'cache':
        if observed != dict(text_encoder=1, video_vae_encode=1, video_vae_decode=0, dit=0):
            raise RuntimeError('Cache encoder call counts differ')
        if Path(case['condition_cache']['path']).parent != out:
            raise RuntimeError('New caption reused an old cache')
        return dict(run=run['complete'], case=case['complete'], audio_guard=case['audio_guard'], condition_cache=case['condition_cache'])
    if observed != dict(text_encoder=0, video_vae_encode=0, video_vae_decode=1, dit=50) or case['sampling']['video_scheduler_updates'] != 50 or case['sampling']['target_audio_generated']:
        raise RuntimeError('Video call counts differ')
    if case['audio_guard']['installed_projection_head_hooks'] != 2 or sha(case['video']['path']) != case['video']['sha256']:
        raise RuntimeError('Video SHA/hook coverage differs')
    sys.path.insert(0, str(SCRIPT.parent))
    from h3_video_infer import verify_media
    media = verify_media(case['video']['path'], run['geometry'])
    return dict(run=run['complete'], case=case['complete'], audio_guard=case['audio_guard'], condition_cache=case['condition_cache'], video=case['video'], independent_media=media, same_image_as_v7=True)


def main():
    if os.getuid() != 0 or not MS.is_dir() or not Path('/dev/shm/h3/Ref2VA').is_dir():
        raise RuntimeError('Host/owner/weights-root preflight failed')
    if sha(BASE) != BASE_SHA or sha(SCRIPT) != SCRIPT_SHA or sha(SCRIPT.parent / 'h3_video_only.py') != WRAPPER_SHA:
        raise RuntimeError('Frozen base/code SHA changed')
    base = read(BASE)
    for config in CONFIGS:
        row = read(config['caption'])
        if row['id'] != SID or row['image_path'] != base['image_path'] or row['caption_version'] != config['version'] or row['motion_caption'] != base['motion_caption'] + ' ' + config['append'] or not row['sidecar']['inference_diagnostic_only'] or not row['sidecar']['formal_SFT_reuse_forbidden'] or row['sidecar']['caption_human_approved']:
            raise RuntimeError('Caption append/fixed-case/diagnostic contract failed')
        for stage in ('cache', 'video'):
            if os.path.lexists(config[stage]):
                raise FileExistsError(config[stage])
    QUEUE.mkdir(exist_ok=False)
    save('plan.json', dict(host='28.216.19.143', manager_pid=os.getpid(), configs=CONFIGS, commands={stage: [command(config, stage) for config in CONFIGS] for stage in ('cache', 'video')}, base_sha256=BASE_SHA, same_case=SID, expected_new_videos=2, base_v7_video_reused=True, formal_SFT_reuse_forbidden=True))
    event('manager_started', identity=process_identity(os.getpid()))
    results = {}
    try:
        for stage in ('cache', 'video'):
            admit(stage)
            running = []
            try:
                for config in CONFIGS:
                    log = (QUEUE / (config['name'] + '_' + stage + '.log')).open('xb')
                    argv = command(config, stage)
                    child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
                    running.append((config, child, log))
                    event('child_started', stage=stage, gpu=config['gpu'], name=config['name'], child_pid=child.pid, argv=argv, log=str(log.name))
                for config, child, log in running:
                    status = child.wait()
                    log.close()
                    event('child_exited', stage=stage, gpu=config['gpu'], name=config['name'], child_pid=child.pid, exit_code=status)
                if any(child.returncode != 0 for _, child, _ in running):
                    raise RuntimeError('Inference child failed; partials retained without overwrite')
            finally:
                for _, child, log in running:
                    if child.poll() is None:
                        child.wait()
                    if not log.closed:
                        log.close()
            results[stage] = {config['name']: validate_stage(config, stage) for config in CONFIGS}
            save(stage + '_verified.json', results[stage])
            event('stage_verified', stage=stage, results=results[stage])
        save('manager_complete.json', dict(complete=True, manager_pid=os.getpid(), results=results, inference_diagnostic_only=True, formal_SFT_reuse_forbidden=True, human_quality_judgment='pending'))
        event('manager_complete', complete=True)
    except BaseException as error:
        event('manager_failed', error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
