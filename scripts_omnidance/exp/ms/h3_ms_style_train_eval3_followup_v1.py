"""Publish only local experiment LoRA exports to shared storage, then evaluate on N4."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import shutil
import subprocess

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
MS = ROOT / 'diff/logs_omnidance/exp/MS'
TRAIN_QUEUE = MS / 'h3_ms_style_sequence100_queue_v3'
LOCAL_TRAIN = Path('/opt/h3_ms_style_sequence100_v3')
STAGED = MS / 'h3_ms_style_sequence100_exports_v3'
EVAL_QUEUE = MS / 'h3_ms_style_sequence100_eval_queue_v3'
CAPTION = MS / 'omni_h3_ms_style_eval3_v1/h3_gt_caption3.json'
CONDITIONS = MS / 'h3_ms_style_eval3_gt_conditions_v1'
INFER = ROOT / 'H3-posttrain/scripts_omnidance/exp/ms/h3_video_infer.py'
INFER_SHA = '77bafd9f6165f80328d4e272c901b21fbf45da3898161bcc039a572e1c86a2f8'
TRAIN_SHA = '247e77eb365685a92680bd3934ac172c7d81242f06fb28f7f4aa665130233c63'
IDS = ['wJPptuhclZmKAEWs', 'wJPrpFsThWAcBQIw', 'wJQTclrlOLEYYfTB']
GROUPS = [('pure_fm', 0.0), ('anchor_0p1', 0.1), ('anchor_1', 1.0)]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def new_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def stage_exports(manager_pid):
    assert MS.is_dir() and not STAGED.exists() and os.geteuid() == 0
    if manager_pid is None:
        raise ValueError('Bind staging to an explicit verified sequence manager PID')
    manager = Path('/proc') / str(manager_pid)
    if manager.exists():
        argv = [value.decode() for value in (manager / 'cmdline').read_bytes().split(b'\0') if value]
        assert manager.stat().st_uid == os.getuid()
        assert str(MS / 'h3_ms_style_sequence100_manager_v1.py') in argv and TRAIN_SHA in argv
        descriptor = os.pidfd_open(manager_pid)
        try:
            print(json.dumps(dict(event='eval3_bound_to_existing_H3_sequence', manager_pid=manager_pid)), flush=True)
            select.select([descriptor], [], [])
        finally:
            os.close(descriptor)
    finished = json.loads((TRAIN_QUEUE / 'finished.json').read_text())
    assert finished['complete'] is True and len(finished['groups']) == 3
    records = []
    for name, anchor in GROUPS:
        result = json.loads((TRAIN_QUEUE / (name + '_finished.json')).read_text())
        assert result['complete'] is True and result['output'] == str(LOCAL_TRAIN / name)
        marker_path = LOCAL_TRAIN / name / 'training-state-final-100/complete.json'
        marker = json.loads(marker_path.read_text())
        assert marker['progress']['optimizer_updates'] == 100 and marker['scope'] == 'source_label_train'
        assert marker['signature']['anchor_lambda'] == anchor
        assert marker['signature']['sources']['code']['h3_video_sft.py'] == TRAIN_SHA
        lora = LOCAL_TRAIN / name / 'step-100.safetensors'
        assert lora.is_file() and 0 < lora.stat().st_size < 2 * 2**30
        records.append(dict(group=name, anchor_lambda=anchor, source_path=str(lora),
            sha256=sha(lora), bytes=lora.stat().st_size,
            complete_state_marker_sha256=sha(marker_path), optimizer_updates=100,
            signature=marker['signature']))
    STAGED.mkdir(exist_ok=False)
    for record in records:
        destination = STAGED / (record['group'] + '-100.safetensors')
        with Path(record['source_path']).open('rb') as src, destination.open('xb') as dst:
            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        assert sha(destination) == record['sha256']
        record['staged_path'] = str(destination)
    new_json(STAGED / 'complete.json', dict(complete=True, groups=records,
        private_shared_experiment_storage=True, public_upload=False,
        full_Adam_or_base_shards_copied=False, source_sha256=sha(__file__)))
    print(json.dumps(dict(event='three100step_native_exports_staged', groups=[r['group'] for r in records])), flush=True)


def evaluate():
    assert os.geteuid() == 0 and MS.is_dir() and not EVAL_QUEUE.exists()
    staged = json.loads((STAGED / 'complete.json').read_text())
    assert staged['complete'] is True and sha(INFER) == INFER_SHA
    assert [row['id'] for row in json.loads(CAPTION.read_text())['records']] == IDS
    assert json.loads((MS / 'h3_ms_style_eval3_gt_queue_v1/finished.json').read_text())['complete'] is True
    cg = Path('/sys/fs/cgroup/memory')
    mem = {line.split()[0][:-1]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    headroom = int((cg / 'memory.limit_in_bytes').read_text()) - int((cg / 'memory.usage_in_bytes').read_text())
    assert headroom >= 180 * 2**30 and mem['MemAvailable'] >= 180 * 2**20
    spec = importlib.util.spec_from_file_location('eval_admission', MS / 'h3_video_dimensions_case1_manager_v8.py')
    admission = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(admission)
    admission.UUIDS = {0: 'GPU-93dc1091-2ba5-13bf-9c46-e716abe867cc'}
    admission.QUEUE = EVAL_QUEUE
    reserve = Path('/tmp/od_reserve_0')
    assert not reserve.exists() or (reserve.is_file() and not reserve.is_symlink() and reserve.stat().st_uid == os.getuid() and reserve.stat().st_size == 0)
    EVAL_QUEUE.mkdir(exist_ok=False)
    new_json(EVAL_QUEUE / 'plan.json', dict(source_code_sha256=sha(__file__),
        infer_source_sha256=INFER_SHA, caption_sha256=sha(CAPTION), staged_marker_sha256=sha(STAGED / 'complete.json'),
        groups=[name for name, _ in GROUPS], ids=IDS, gpu=0,
        original_GT_baselines_reused=True, additional_videos=9, quality_accepted=False,
        headroom_bytes=headroom, memavailable_bytes=mem['MemAvailable'] * 1024))
    reserve.touch()
    admission.admit('three_H3_100step_video_evaluation')
    reports = []
    for record in staged['groups']:
        assert sha(record['staged_path']) == record['sha256'] and sha(INFER) == INFER_SHA
        out = MS / ('h3_ms_style_' + record['group'] + '_s100_eval3_v3')
        assert not out.exists()
        command = ['env', 'CUDA_VISIBLE_DEVICES=0', 'OMP_NUM_THREADS=2', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
            '/opt/venv_h3/bin/python', '-B', '-u', str(INFER), '--caption-json', str(CAPTION),
            '--condition-cache', str(CONDITIONS), '--weights-root', '/dev/shm/h3/Ref2VA',
            '--out', str(out), '--limit', '3', '--lora', record['staged_path']]
        new_json(EVAL_QUEUE / (record['group'] + '_plan.json'), dict(command=command))
        with (EVAL_QUEUE / (record['group'] + '.log')).open('xb') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            new_json(EVAL_QUEUE / (record['group'] + '_live.json'), dict(child_pid=child.pid,
                manager_pid=os.getpid(), started_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            while True:
                try:
                    code = child.wait(timeout=300)
                    break
                except subprocess.TimeoutExpired:
                    reserve.touch()
        assert code == 0, 'Generation failed; no duplicate launch or partial overwrite'
        run = json.loads((out / 'run_complete.json').read_text())
        assert run['complete'] is True and run['completed_ids'] == IDS
        assert run['lora_source']['file']['sha256'] == record['sha256']
        reports.append(dict(group=record['group'], out=str(out), complete=True, generated_videos=3))
    new_json(EVAL_QUEUE / 'finished.json', dict(complete=True, groups=reports,
        fixed_cases=IDS, actualhuman=False, human_quality_accepted=False,
        audio_input=False, audio_output=False, source_sha256=sha(__file__)))
    print(json.dumps(dict(event='three_H3_groups_fixedcase_evaluation_complete', groups=reports)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['stage', 'evaluate'])
    parser.add_argument('--manager-pid', type=int)
    args = parser.parse_args()
    stage_exports(args.manager_pid) if args.mode == 'stage' else evaluate()
