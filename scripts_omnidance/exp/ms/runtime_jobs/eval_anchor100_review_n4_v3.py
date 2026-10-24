"""N4 fixed-three anchor100 inference after independent state/source/LoRA audit."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
MS = ROOT / 'diff/logs_omnidance/exp/MS'
OUT = MS / 'review_h3_anchor01_s100_n4_v3'
AUDIT = MS / 'h3_anchor100_independent_audit_v4_20261024'
INFER = ROOT / 'H3-posttrain/scripts_omnidance/exp/ms/h3_video_infer_chunked_lora_v1.py'
CORE = INFER.with_name('h3_video_infer.py')
INFER_SHA = 'c611d16c26c38e918d4c91560f7fa72e6e88ad93b43ecad1b48fd80a91deb665'
CORE_SHA = '77bafd9f6165f80328d4e272c901b21fbf45da3898161bcc039a572e1c86a2f8'
SOURCE_SHA = '247e77eb365685a92680bd3934ac172c7d81242f06fb28f7f4aa665130233c63'
WRAPPER_SHA = '89439d733d62f522bb3dd993bc456541b3ca15ead34ff2e19f86657a471c5055'
GT = MS / 'omni_h3_ms_style_eval3_v1/h3_gt_caption3.json'
GT_SHA = '17ff545eff693e19fd18fb60f10f6ae452636e14a0f06a0b6a138e0cdbad8020'
CACHE = MS / 'h3_ms_style_eval3_gt_conditions_v1'
IDS = ['wJPptuhclZmKAEWs', 'wJPrpFsThWAcBQIw', 'wJQTclrlOLEYYfTB']
HOST = '28.216.19.143'
GPU = 2
GPU_UUID = 'GPU-f9c8016f-56cb-83c2-7230-f3d4868e9264'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def event(name, **value):
    with (OUT / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(event=name, manager_pid=os.getpid(),
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **value)) + '\n')
        stream.flush()


def identity(pid):
    p = Path('/proc') / str(pid)
    return dict(pid=pid, argv=[v.decode() for v in (p / 'cmdline').read_bytes().split(bytes([0])) if v],
         start_ticks=(p / 'stat').read_text().rsplit(')', 1)[1].split()[19])


def frozen(manager_sha):
    assert sha(__file__) == manager_sha
    for path, expected in [(INFER, INFER_SHA), (CORE, CORE_SHA), (GT, GT_SHA),
            (CORE.with_name('h3_video_sft.py'), SOURCE_SHA), (CORE.with_name('h3_video_only.py'), WRAPPER_SHA)]:
        assert sha(path) == expected, 'Frozen source changed: ' + str(path)


def training():
    terminal = json.loads((AUDIT / 'finished.json').read_text())
    assert terminal['complete'] is True and terminal['full_checkpoint_reverified'] is True
    report = json.loads((AUDIT / 'full_state_verified.json').read_text())
    for key in ['complete', 'full_checkpoint_reverified', 'independent_audit', 'typed_numeric_verified',
                'all_file_sha256_verified', 'model_and_adam_and_rng_and_scheduler_finite',
                'lora_all_ab_source_verified', 'fm_recipe_verified']:
        assert report[key] is True, 'Required independent full-state check missing: ' + key
    for key, expected in {'actual_training_updates': 100, 'world_size': 8, 'lora_rank': 32,
            'lora_alpha': 64, 'learning_rate': 1e-5, 'anchor_lambda': .1, 'training_cfg_scale': 1.,
            'complete_state_members': 28}.items():
        assert type(report[key]) in (int, float) and report[key] == expected
    assert report['source_sha256'] == SOURCE_SHA and report['wrapper_sha256'] == WRAPPER_SHA
    assert terminal['lora_sha256'] == report['lora_sha256']
    assert report['marker_sha256'] == sha(Path(report['state']) / 'complete.json')
    assert report['lora_sha256'] == sha(report['lora'])
    return report


def admission(approval):
    assert approval.is_file() and not approval.is_symlink()
    receipt = json.loads(approval.read_text())
    assert receipt['complete'] is True and receipt['host'] == HOST and receipt['gpu'] == GPU
    assert receipt['gpu_uuid'] == GPU_UUID and receipt['resource_owner'] == 'goal-live-pipelines-2'
    reserve = Path('/tmp/od_reserve_2')
    assert reserve.is_file() and not reserve.is_symlink() and reserve.stat().st_uid == 0 and reserve.stat().st_size == 0
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()
    gpu = [[v.strip() for v in line.split(',')] for line in rows if line.split(',')[0].strip() == str(GPU)][0]
    assert gpu[:2] == [str(GPU), GPU_UUID] and float(gpu[2]) <= 1200 and float(gpu[3]) >= 78000
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True).splitlines()
    assert not [line for line in apps if line.split(',')[0].strip() == GPU_UUID], 'Owner must release GPU2 before this manager; never terminate holders here'
    cg = Path('/sys/fs/cgroup/memory')
    room = int((cg / 'memory.limit_in_bytes').read_text()) - int((cg / 'memory.usage_in_bytes').read_text())
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))) * 1024
    assert min(room, available) >= 180 * 2**30
    save(OUT / 'admission_verified.json', dict(complete=True, approval=str(approval), approval_sha256=sha(approval),
        approval_receipt=receipt, gpu=gpu, compute_apps=apps, cpu_headroom_bytes=room, available_bytes=available,
        holder_terminated_by_this_manager=False))
    return reserve


def decode_all(videos):
    code = """import decord,json,sys; from pathlib import Path
ids=json.loads(sys.argv[1]); root=Path(sys.argv[2]); results=[]
for sid in ids:
    path=root/(sid+'.mp4'); reader=decord.VideoReader(str(path),ctx=decord.cpu(0),num_threads=2)
    assert len(reader)==124 and abs(reader.get_avg_fps()-24)<1e-4
    for index in range(len(reader)):
        frame=reader[index].asnumpy(); assert frame.shape==(1280,704,3)
    results.append(dict(id=sid,path=str(path),frames=124,width=704,height=1280,fps=reader.get_avg_fps(),complete_decode=True,decoder='decord CPU every frame'))
print(json.dumps(dict(complete=True,ids=ids,records=results)))
"""
    result = subprocess.run(['/opt/venv_eval/bin/python', '-B', '-c', code, json.dumps(IDS), str(videos)],
        capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def video_audit(report):
    videos = OUT / 'videos'
    run = json.loads((videos / 'run_complete.json').read_text())
    assert run['complete'] is True and run['completed_ids'] == IDS and run['mode'] == 'video_generation'
    assert run['geometry'] == dict(height=1280, width=704, num_frames=124, fps=24)
    assert (run['steps'], run['seed'], run['cfg'], run['flow_shift']) == (50, 42, 1., 12.)
    assert run['lora_source']['file']['sha256'] == report['lora_sha256'] and run['lora_source']['alpha'] == 1.
    assert run['source_code']['sha256'] == CORE_SHA and run['wrapper_source']['sha256'] == WRAPPER_SHA
    assert run['caption_source']['sha256'] == GT_SHA
    original_dit = json.loads((Path(report['state']) / 'complete.json').read_text())['signature']['sources']['original_dit']
    observed = {str(Path(item['path']).relative_to(run['model_sources']['weights_root'])): item
                for item in run['model_sources']['files']}
    transformer_names = {name for name in original_dit if name.startswith('transformer/')}
    assert transformer_names == {name for name in observed if name.startswith('transformer/')}
    for name in transformer_names:
        assert observed[name]['sha256'] == original_dit[name]['sha256']
        assert observed[name]['size_bytes'] == original_dit[name]['bytes']
    model_index = Path(run['model_sources']['weights_root']) / 'model_index.json'
    assert sha(model_index) == original_dit['model_index.json']['sha256']
    runtime = json.loads((videos / 'lora_chunk_runtime_verified.json').read_text())
    assert runtime['complete'] is True and runtime['source_sha256'] == INFER_SHA
    assert runtime['actual_call_counts']['layer_calls'] > 0 and runtime['actual_call_counts']['chunk_calls'] > 0
    cases = []
    for sid in IDS:
        audit = json.loads((videos / (sid + '.audit.json')).read_text())
        media = audit['media']
        assert audit['complete'] is True and audit['id'] == sid and media['complete_decode'] is True
        assert (media['frames'], media['width'], media['height'], media['fps'], media['audio_streams']) == (124, 704, 1280, 24., 0)
        assert not any(audit['audio_guard']['calls'].values()) and audit['sampling']['video_scheduler_updates'] == 50
        assert audit['runtime_lora_implementation']['source_sha256'] == INFER_SHA
        assert audit['video']['sha256'] == sha(videos / (sid + '.mp4'))
        assert audit['condition_cache']['sha256'] == sha(CACHE / (sid + '.condition.pt'))
        cases.append(dict(id=sid, path=str(videos / (sid + '.mp4')), video_sha256=audit['video']['sha256'],
            audit=str(videos / (sid + '.audit.json')), audit_sha256=sha(videos / (sid + '.audit.json'))))
    decoded = decode_all(videos)
    save(OUT / 'decord_full_decode_verified.json', decoded)
    save(OUT / 'full_video_verified.json', dict(complete=True, ids=IDS, cases=cases, lora_sha256=report['lora_sha256'],
        infer_source_sha256=INFER_SHA, core_source_sha256=CORE_SHA, training_source_sha256=SOURCE_SHA,
        independent_training_report=str(AUDIT / 'full_state_verified.json'), independent_training_report_sha256=sha(AUDIT / 'full_state_verified.json'),
        decord_full_decode_verified=True, original_dit_identity_verified=True,
        original_dit_transformer_files=len(transformer_names), internal_audio=False, human_quality_accepted=False))
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--manager-sha', required=True)
    parser.add_argument('--resource-approval', type=Path)
    parser.add_argument('--cpu-preflight-only', action='store_true')
    args = parser.parse_args()
    assert str(uuid.UUID(args.run_id)) == args.run_id and os.getuid() == 0
    frozen(args.manager_sha)
    assert [r['id'] for r in json.loads(GT.read_text())['records']] == IDS
    assert json.loads((CACHE / 'run_complete.json').read_text())['completed_ids'] == IDS
    if args.cpu_preflight_only:
        print(json.dumps(dict(complete=True, cpu_preflight_only=True, gpu_forward_performed=False, ids=IDS)))
        return
    assert args.resource_approval is not None, 'Explicit owner admission receipt required'
    assert not OUT.exists(), 'Do not overwrite old or duplicate outputs'
    OUT.mkdir()
    save(OUT / 'manager_live.json', dict(identity=identity(os.getpid()), run_id=args.run_id,
         source_sha256=args.manager_sha, host=HOST, gpu=GPU, uuid=GPU_UUID))
    try:
        report = training()
        save(OUT / 'training_verified_receipt.json', report)
        lora = Path('/opt/h3_review_anchor01_step100_v3_' + args.run_id + '.safetensors')
        assert not lora.exists()
        subprocess.run(['scp', '-O', '-p', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
             '28.216.19.137:' + report['lora'], str(lora)], check=True)
        assert sha(lora) == report['lora_sha256']
        save(OUT / 'weight_verified.json', dict(lora=str(lora), original_lora=report['lora'], sha256=report['lora_sha256'],
             actual_training_updates=100, tensor_keys=report['lora_tensor_keys'], full_ab_source_verified=True))
        reserve = admission(args.resource_approval)
        argv = ['/opt/venv_h3/bin/python', '-B', '-u', str(INFER), '--caption-json', str(GT),
             '--weights-root', '/dev/shm/h3/Ref2VA', '--limit', '3', '--steps', '50', '--cfg', '1', '--seed', '42',
             '--flow-shift', '12', '--out', str(OUT / 'videos'), '--condition-cache', str(CACHE), '--lora', str(lora)]
        save(OUT / 'plan.json', dict(command=argv, ids=IDS, gpu=GPU, uuid=GPU_UUID, cache=str(CACHE),
             cache_reused=True, internal_audio=False, human_quality_accepted=False))
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(GPU), OMP_NUM_THREADS='2', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
             HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', DIFFSYNTH_SKIP_DOWNLOAD='True')
        with (OUT / 'video.log').open('x') as stream:
            frozen(args.manager_sha)
            child = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT)
            save(OUT / 'video_live.json', dict(manager_pid=os.getpid(), child_pid=child.pid, identity=identity(child.pid),
                command=argv, host=HOST, gpu=GPU, uuid=GPU_UUID))
            event('actual_anchor100_video_started', child_pid=child.pid, gpu=GPU, actual_training_updates=100)
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    reserve.touch()
        save(OUT / 'video_exit.json', dict(exit_code=code))
        assert code == 0, 'Video stage failed; preserve partial output without retry'
        frozen(args.manager_sha)
        cases = video_audit(report)
        save(OUT / 'finished.json', dict(complete=True, ids=IDS, actual_training_updates=100, cases=cases,
            decord_full_decode_verified=True, full_checkpoint_reverified=True, human_quality_accepted=False))
        event('anchor100_three_videos_full_verified', ids=IDS)
    except Exception as error:
        save(OUT / 'finished.json', dict(complete=False, error=repr(error), partial_outputs_preserved=True, retry_started=False))
        event('anchor100_v3_blocked_preserve_output', error=repr(error))
        raise


if __name__ == '__main__':
    main()
