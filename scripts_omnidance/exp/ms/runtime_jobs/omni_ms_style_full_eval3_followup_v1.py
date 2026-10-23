"""Evaluate the final full-data model on its own node after training completes."""
import datetime
import importlib.util
import json
import os
from pathlib import Path
import select
import subprocess

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
MS = ROOT / 'diff/logs_omnidance/exp/MS'
TRAIN_QUEUE = MS / 'omni_ms_style_full_queue_v4'
TRAIN = Path('/opt/omni_ms_style_full_train_v3')
SOURCE = ROOT / 'H3-posttrain/scripts_omnidance/exp/ms/omni_dance_sft.py'
SOURCE_SHA = '4d82caffd4f6ce354fd6fbd240827d0cac69d66414de33e847501369b0e82dd2'
QUEUE = MS / 'omni_ms_style_full_eval3_queue_v2'
METADATA = MS / 'omni_h3_ms_style_eval3_v1/omni_infer3.jsonl'
MODEL = Path('/opt/omni30b_dance_official_v1')
IDS = ['wJPptuhclZmKAEWs', 'wJPrpFsThWAcBQIw', 'wJQTclrlOLEYYfTB']


def main():
    assert os.geteuid() == 0 and MS.is_dir() and not QUEUE.exists()
    spec = importlib.util.spec_from_file_location('full_eval_admission', MS / 'omni_ms_style_pilot_manager_v1.py')
    admission = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(admission)
    assert admission.sha(SOURCE) == SOURCE_SHA
    parent_plan = json.loads((TRAIN_QUEUE / 'plan.json').read_text())
    assert parent_plan['output'] == str(TRAIN)
    parent_live = json.loads((TRAIN_QUEUE / 'live.json').read_text())
    QUEUE.mkdir(exist_ok=False)
    admission.QUEUE = QUEUE
    admission.save('plan.json', dict(node='N7', gpu=0, source_sha256=SOURCE_SHA,
        parent_queue=str(TRAIN_QUEUE), parent_training=str(TRAIN), fixed_ids=IDS,
        parent_manager_pid=parent_live['manager_pid'], metadata_sha256=admission.sha(METADATA),
        reuse_original_and_pilot_comparison=True, no_new_training=True,
        generation_policy='Frozen production inference, deterministic,512token cap,no added penalties or output cleaning',
        quality_accepted=False))
    process = Path('/proc') / str(parent_live['manager_pid'])
    if process.exists():
        argv = [value.decode() for value in (process / 'cmdline').read_bytes().split(b'\0') if value]
        assert process.stat().st_uid == os.getuid()
        assert str(MS / 'omni_ms_style_full_manager_v2.py') in argv and parent_plan['entry_sha256'] in argv
        descriptor = os.pidfd_open(parent_live['manager_pid'])
        try:
            print(json.dumps(dict(event='full_eval3_bound_to_running6000', manager_pid=parent_live['manager_pid'])), flush=True)
            select.select([descriptor], [], [])
        finally:
            os.close(descriptor)
    done = json.loads((TRAIN_QUEUE / 'finished.json').read_text())
    result = json.loads((TRAIN / 'complete.json').read_text())
    assert done['complete'] is True and result['data_scope'] == 'source_label_full'
    assert 3 <= result['full_epochs'] <= 10 and result['updates'] == 750 * result['full_epochs']
    checkpoint = TRAIN / result['checkpoint']
    marker = json.loads((checkpoint / 'complete.json').read_text())
    assert marker['fingerprint'] == result['fingerprint'] and marker['updates'] == result['updates']
    own_cgroup = Path('/proc/self/cgroup').read_text()
    cg = Path('/sys/fs/cgroup/memory')
    headroom = int((cg / 'memory.limit_in_bytes').read_text()) - int((cg / 'memory.usage_in_bytes').read_text())
    mem = {line.split()[0][:-1]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    assert headroom >= 180 * 2**30 and mem['MemAvailable'] >= 180 * 2**20
    admission.save('memory_admission.json', dict(cgroup_headroom_bytes=headroom,
        host_MemAvailable_bytes=mem['MemAvailable'] * 1024, threshold_bytes=180 * 2**30,
        previous_failed_queue=str(MS / 'omni_ms_style_full_eval3_queue_v1'),
        previous_failure='Host/cgroup memory admission before model loading; original queue preserved.'))
    spec2 = importlib.util.spec_from_file_location('single_gpu_admission', MS / 'dance_caption_observed32_30b_manager_v1.py')
    single = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(single)
    before = single.snapshot()
    assert len(before['processes']) <= 1
    reserve = Path('/tmp/od_reserve_0')
    assert not reserve.exists() or (reserve.is_file() and not reserve.is_symlink() and reserve.stat().st_uid == os.getuid() and reserve.stat().st_size == 0)
    reserve.touch()
    import signal
    for row in before['processes']:
        value = row['identity']
        descriptor = os.pidfd_open(value['pid'])
        try:
            assert value['uids'] == [0] * 4 and value['argv'] == single.HOLDER and value['CUDA_VISIBLE_DEVICES'] == '0'
            assert row['used_MiB'] <= 1100 and admission.identity(value['pid'])['cgroup'] == own_cgroup
            assert single.identity(value['pid']) == value and single.snapshot()['processes'] == [row]
            signal.pidfd_send_signal(descriptor, signal.SIGTERM)
            signal.pidfd_send_signal(descriptor, signal.SIGCONT)
            assert select.select([descriptor], [], [], 30)[0], 'Exact holder did not exit'
        finally:
            os.close(descriptor)
    assert not single.snapshot()['processes'] and admission.sha(SOURCE) == SOURCE_SHA
    out = QUEUE / 'full_predictions.json'
    command = ['env', 'CUDA_VISIBLE_DEVICES=0', 'OMP_NUM_THREADS=2', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
        '/opt/venv_h3/bin/python', '-B', '-u', str(SOURCE), 'infer', '--metadata', str(METADATA),
        '--model', str(MODEL), '--model-id', 'Qwen/Qwen3-Omni-30B-A3B-Instruct@26291f793822fb6be9555850f06dfe95f2d7e695',
        '--checkpoint', str(checkpoint), '--out', str(out), '--attention', 'sdpa', '--max-new-tokens', '512', '--ids', *IDS]
    admission.save('command.json', dict(command=command, checkpoint_marker_sha256=admission.sha(checkpoint / 'complete.json')))
    with (QUEUE / 'inference.log').open('xb') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        admission.save('live.json', dict(manager_pid=os.getpid(), child_pid=child.pid,
            started_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        while True:
            try:
                code = child.wait(timeout=300)
                break
            except subprocess.TimeoutExpired:
                reserve.touch()
    complete = code == 0 and out.is_file()
    if complete:
        predictions = json.loads(out.read_text())['predictions']
        assert [row['id'] for row in predictions] == IDS
    admission.save('finished.json', dict(complete=complete, exit_code=code, fixed_ids=IDS,
        predictions=str(out), raw_caption_quality_verified=False, public_upload=False,
        no_GT_in_user_input=True, actual_training_added=False))
    print(json.dumps(dict(event='full_eval3_finished', complete=complete, exit_code=code, predictions=str(out))), flush=True)
    raise SystemExit(code or (0 if complete else 1))


if __name__ == '__main__':
    main()
