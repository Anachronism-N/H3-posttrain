"""Foreground N4 fixed-three complete mean/token native video acceptance."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
from music_direct_compare_manager_v3 import identity, snapshot, sha, HOLDER

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
REPO = ROOT / 'H3-posttrain'
MS = ROOT / 'diff/logs_omnidance/exp/MS'
OUT = MS / 'music_direct_video_compare_n4_v1'
JOINT = MS / 'music_direct_eval3_generation_joint_v1.json'
IDS = ['wJPptuhclZmKAEWs', 'wJPrpFsThWAcBQIw', 'wJQTclrlOLEYYfTB']
ARMS = [('mean', 0, 'GPU-93dc1091-2ba5-13bf-9c46-e716abe867cc',
         MS / 'music_direct_mean100_n2_v3/last.pt',
         '417d64d2d7cbee3d44f36b82c4eaf192c3959caf9aface9e6411d4e9aef78a82'),
        ('token', 1, 'GPU-f940954a-c4f3-f1b4-25fe-9c2ad89a9e93',
         MS / 'music_direct_token100_n3_v3/last.pt',
         'f8350e14b44a882a02fb47c784c5875fd145e1dac94b5d35cfb44d1365f68473')]
ENTRY = REPO / 'scripts_omnidance/exp/h3ar/sample_h3_ar.py'
UPSTREAM = ROOT / 'DiffSynth-Studio-upstream-20260930'
WEIGHTS = Path('/dev/shm/h3/Ref2VA')
CANONICAL = ROOT / 'hyperflow_exp/weights/minimax-h3/Ref2VA'


def save(path, record):
    with Path(path).open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)


def environment(gpu=None):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONPATH=str(UPSTREAM), PYTHONUNBUFFERED='1')
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return env


def command(arm, checkpoint):
    return ['/opt/venv_h3/bin/python', str(ENTRY), '--out', str(OUT / arm),
            '--cached_prefix_manifest', str(JOINT), '--direct_music_ckpt', str(checkpoint),
            '--conds', 'joint', '--rows', '3', '--mode', 'bi', '--steps', '50',
            '--cfg', '1', '--seed', '42', '--height', '1280', '--width', '704',
            '--num_frames', '124', '--shift', '12', '--audio_shift', '3',
            '--no_text_encoder', '--w', str(WEIGHTS), '--music_mode', 'ref',
            '--output_audio', 'original_mux']


def edge_fingerprint(path):
    n = 1024 * 1024
    size = path.stat().st_size
    with path.open('rb') as stream:
        head = stream.read(n)
        stream.seek(max(0, size - n))
        tail = stream.read(n)
    return dict(size=size, edge_sha256=hashlib.sha256(head + tail).hexdigest())


def preflight():
    from music_text_oracle_generate import (validate_direct_generation_manifest,
                                           direct_native_parser, direct_sampling_settings)
    manifest, manifest_sha = validate_direct_generation_manifest(JOINT)
    assert manifest['condition'] == 'joint' and [c['sample_id'] for c in manifest['cases']] == IDS
    split_source = manifest['source_files']['split']
    split = json.loads(Path(split_source['path']).read_text())
    assert split['train'] == [] and split['val'] == IDS
    assert Path('/opt/venv_music/bin/python').is_file()
    files = [Path(__file__), ENTRY, Path(__file__).with_name('music_text_oracle_generate.py'),
             Path(__file__).with_name('music_text_infer.py'), Path(__file__).with_name('music_text_adapter.py'),
             Path(__file__).with_name('music_text_condition.py'),
             UPSTREAM / 'diffsynth/pipelines/minimax_h3_audio_video.py',
             UPSTREAM / 'diffsynth/utils/data/audio_video.py']
    source = {str(p): sha(p) for p in files}
    base = []
    for relative in ['model_index.json', 'transformer/config.json', 'transformer/model.safetensors.index.json']:
        a, b = WEIGHTS / relative, CANONICAL / relative
        assert sha(a) == sha(b), ('Base metadata mismatch', relative)
        base.append(dict(relative=relative, sha256=sha(a)))
    weight_files = sorted((WEIGHTS / 'transformer').glob('model-*.safetensors'))
    assert len(weight_files) == 13
    weight_files += [WEIGHTS / 'video_vae/source/model.safetensors', WEIGHTS / 'audio_vae/model.safetensors']
    for path in weight_files:
        relative = path.relative_to(WEIGHTS)
        a, b = edge_fingerprint(path), edge_fingerprint(CANONICAL / relative)
        assert a == b, ('Base weight edge identity mismatch', str(relative))
        base.append(dict(relative=str(relative), **a))
    scopes = []
    for arm, gpu, uuid, checkpoint, checksum in ARMS:
        assert sha(checkpoint) == checksum
        cli = command(arm, checkpoint)
        args = direct_native_parser().parse_args(cli[2:])
        assert args.rows == 3 and args.conds == 'joint' and args.no_text_encoder
        assert (args.steps, args.shift, args.audio_shift, args.seed, args.cfg) == (50, 12, 3, 42, 1)
        assert direct_sampling_settings(args, args.height, args.width)['num_frames'] == 124
        code = """import sys,json,torch,gc
sys.path.insert(0,sys.argv[1])
from music_text_adapter import load_direct_music_text_checkpoint
model=load_direct_music_text_checkpoint(sys.argv[2]).eval()
assert len(model.state_dict())==467
assert not model.checkpoint_metadata['freeze_backbone']
assert model.checkpoint_metadata['context_policy']=='forbidden'
print(json.dumps(dict(strict_complete_state=True,state_tensors=len(model.state_dict()),metadata=model.checkpoint_metadata)))
"""
        result = subprocess.run(['/opt/venv_music/bin/python', '-c', code, str(Path(__file__).parent), str(checkpoint)],
                                env=environment(), text=True, capture_output=True, timeout=300)
        if result.returncode:
            raise RuntimeError('CPU complete loader failed: ' + result.stderr[-6000:] + result.stdout[-3000:])
        scopes.append(dict(arm=arm, gpu=gpu, gpu_uuid=uuid, checkpoint=str(checkpoint),
                           checkpoint_sha256=checksum, command=cli, strict_loader_stdout=result.stdout))
    return dict(complete=True, fixed_case_ids=IDS, manifest_sha256=manifest_sha, source_files=source,
                strict_checkpoint_scopes=scopes, base_identity=base,
                base_identity_method='exact canonical metadata and all 15 weight-file sizes plus 1MiB head/tail fingerprints; not a full-weight SHA scan',
                teacher_prefix_model_weight_content_sha256=manifest['caches'][0]['model_weight_content_sha256'],
                source_people=3, expected_videos=6, no_training=True), manifest


def acquire():
    resource = snapshot()
    assert resource['memavailable_bytes'] >= 360 * 1024**3
    assert resource['cgroup_headroom_bytes'] >= 360 * 1024**3
    leases, holders = [], []
    for arm, gpu, uuid, checkpoint, checksum in ARMS:
        selected = next(r for r in resource['gpus'] if r[0] == str(gpu))
        assert selected[1] == uuid
        lease = open('/tmp/od_direct_video_n4_gpu' + str(gpu) + '.lock', 'a')
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        leases.append(lease)
        reserve = Path('/tmp/od_reserve_' + str(gpu))
        assert not reserve.exists() or (reserve.is_file() and not reserve.is_symlink()
               and reserve.stat().st_uid == 0 and reserve.stat().st_size == 0)
        active = [r for r in resource['apps'] if r[1] == uuid]
        assert len(active) <= 1, ('Multiple GPU processes', active)
        if active:
            pid = int(active[0][0])
            holder = identity(pid)
            assert holder['uids'] == [0]*4 and len(holder['argv']) == 2
            assert holder['argv'][1] == HOLDER and holder['cuda'] == str(gpu)
            assert holder['cgroup'] == resource['cgroup']
            assert float(active[0][2]) < 1200, ('Holder GPU memory unexpected', active)
            holders.append(holder)
    # Validate every selected process before sending any signal.
    for holder in holders:
        fd = os.pidfd_open(holder['pid'])
        try:
            assert identity(holder['pid']) == holder
            signal.pidfd_send_signal(fd, signal.SIGTERM)
            assert select.select([fd], [], [], 20)[0], ('Holder did not exit', holder['pid'])
        finally:
            os.close(fd)
    for _, gpu, uuid, _, _ in ARMS:
        Path('/tmp/od_reserve_' + str(gpu)).touch()
        assert not [r for r in snapshot()['apps'] if r[1] == uuid], 'GPU process appeared after admission'
    save(OUT / 'admission.json', dict(complete=True, before=resource, released_holders=holders))
    return leases


def validate_arm(arm, manifest):
    from music_generation_validate import probe_video
    records = []
    for case in manifest['cases']:
        output = OUT / arm / 'joint' / (case['sample_id'] + '_' + Path(case['audio_path']).stem + '.mp4')
        record = json.loads(output.with_suffix('.audit.json').read_text())
        audit, inputs = record['audit'], record['input']
        assert record['complete'] and inputs['case'] == case
        assert audit['scheduler_updates'] == dict(video=50, audio=50)
        assert audit['full_cached_prefix_preserved'] and audit['original_cached_prefix_tags_preserved']
        assert audit['packed_text_tags_exact'] and audit['native_audio_vae_unchanged'] and audit['original_joint_pipeline']
        assert audit['native_audio_condition'] == 'real_reference_audio' and audit['music_mode'] == 'ref'
        assert audit['reference']['exists'] and audit['visual_reference']['exists']
        assert audit['text_encoder_calls'] == 0 and audit['semantic_text'] is None
        assert audit['prompt_embeds_shape'] == [case['prefix_length'] + 254, 5120]
        assert not inputs['student_reads_target_or_caption'] and not inputs['teacher_identity'] and not inputs['student_zero_input']
        assert inputs['output_audio'] == 'original_mux' and record['mux_audio_sha256'] == inputs['native_waveform_sha256']
        assert record['video_sha256'] == sha(output)
        records.append(dict(arm=arm, sample_id=case['sample_id'], path=str(output), sha256=sha(output),
                            audit_path=str(output.with_suffix('.audit.json')), **probe_video(output)))
    save(OUT / arm / 'validated.json', dict(complete=True, expected=3, cases=records,
                                         full_media_decode_checked=True, human_quality_pending=True))
    return records


def run_arm(arm_info, manifest):
    arm, gpu, uuid, checkpoint, checksum = arm_info
    with (OUT / 'logs' / (arm + '.log')).open('x') as stream:
        process = subprocess.Popen(command(arm, checkpoint), stdout=stream, stderr=subprocess.STDOUT, env=environment(gpu))
        save(OUT / arm / 'child.json', dict(pid=process.pid, identity=identity(process.pid), gpu=gpu, gpu_uuid=uuid))
        print(json.dumps(dict(event='child_started', arm=arm, gpu=gpu, gpu_uuid=uuid, pid=process.pid)), flush=True)
        while process.poll() is None:
            Path('/tmp/od_reserve_' + str(gpu)).touch()
            time.sleep(20)
        exit_code = process.returncode
    save(OUT / arm / 'exit.json', dict(exit_code=exit_code, pid=process.pid))
    if exit_code:
        raise RuntimeError(arm + ' sampler exited ' + str(exit_code))
    text = (OUT / 'logs' / (arm + '.log')).read_text()
    assert '[h3ar] done' in text and text.count('[direct native saved] ') == 3
    assert 'Traceback (most recent call last)' not in text and 'LoRA' not in text
    assert sha(checkpoint) == checksum
    return validate_arm(arm, manifest)


def main():
    assert os.getuid() == 0 and not OUT.exists(), 'Require UID0 and a fresh output tree'
    checked, manifest = preflight()
    OUT.mkdir()
    (OUT / 'logs').mkdir()
    for arm, *_ in ARMS:
        (OUT / arm).mkdir()
        (OUT / arm / 'joint').mkdir()
    save(OUT / 'cpu_preflight.json', checked)
    print(json.dumps(dict(event='cpu_preflight_complete', out=str(OUT))), flush=True)
    leases = acquire()
    save(OUT / 'manager.json', dict(pid=os.getpid(), identity=identity(os.getpid()), host='28.216.19.143'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_arm, arm, manifest) for arm in ARMS]
        records = [record for future in futures for record in future.result()]
    assert sha(JOINT) == checked['manifest_sha256']
    assert all(sha(path) == digest for path, digest in checked['source_files'].items())
    save(OUT / 'report.json', dict(complete=True, expected=6, source_people=3, cases=records,
        full_media_decode_checked=True, human_quality_pending=True, winner=None,
        no_text_encoder=True, cached_full_prefix_and_tags_preserved=True,
        native_audio='real_reference_audio_original_joint_50_video_and_audio_steps_original_mux',
        preflight_sha256=sha(OUT / 'cpu_preflight.json')))
    print(json.dumps(dict(event='six_videos_validated', report=str(OUT / 'report.json'))), flush=True)


if __name__ == '__main__':
    main()
