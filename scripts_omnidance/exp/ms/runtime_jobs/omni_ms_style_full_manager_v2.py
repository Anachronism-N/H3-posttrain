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
ENTRY = MS / 'omni_ms_style_full_train_v2.py'
CORE = ROOT / 'H3-posttrain/scripts_omnidance/exp/ms/omni_dance_sft.py'
CORE_SHA = '4d82caffd4f6ce354fd6fbd240827d0cac69d66414de33e847501369b0e82dd2'
DATA = MS / 'omni_ms_style_full6500_v2/omni.jsonl'
QUEUE = MS / 'omni_ms_style_full_queue_v4'
OUT = Path('/opt/omni_ms_style_full_train_v3')
PILOT_QUEUE = MS / 'omni_ms_style_pilot_queue_v1'
PILOT_OUT = MS / 'omni_ms_style_pilot_train_v1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry-sha', required=True)
    parser.add_argument('--metadata-sha', required=True)
    args = parser.parse_args()
    assert os.geteuid() == 0 and MS.is_dir() and OUT.parent.is_dir() and not OUT.exists()
    assert not QUEUE.exists()
    spec = importlib.util.spec_from_file_location('trusted_pilot_admission', MS / 'omni_ms_style_pilot_manager_v1.py')
    admission = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(admission)
    assert admission.sha(ENTRY) == args.entry_sha and admission.sha(CORE) == CORE_SHA
    assert admission.sha(DATA) == args.metadata_sha
    QUEUE.mkdir(exist_ok=False)
    admission.QUEUE = QUEUE
    admission.save('plan.json', dict(host='28.216.18.144', entry=str(ENTRY), entry_sha256=args.entry_sha,
        core_sha256=CORE_SHA, metadata=str(DATA), metadata_sha256=args.metadata_sha,
        counts=dict(train=6000, val=500), min_epochs=3, max_epochs=10,
        data_scope='source_label_full', semantic_approval=False, output=str(OUT),
        checkpoint_storage='Existing node-local disk; full optimizer state remains private and recoverable.',
        predecessor=str(PILOT_OUT), fresh_official_base=True, optimizer_resume=False,
        authorization='User requested existing MS+style full6000/500 and parallel preparation on 2026-10-23'))
    live = json.loads((PILOT_QUEUE / 'live.json').read_text())
    previous_pid = live['manager_pid']
    root = Path('/proc') / str(previous_pid)
    if root.exists():
        argv = [value.decode() for value in (root / 'cmdline').read_bytes().split(b'\0') if value]
        assert root.stat().st_uid == os.getuid() and str(MS / 'omni_ms_style_pilot_manager_v1.py') in argv
        assert CORE_SHA in argv, 'Unknown predecessor manager identity'
        descriptor = os.pidfd_open(previous_pid)
        try:
            print(json.dumps(dict(event='full6000_waiting_existing_pilot', predecessor_pid=previous_pid)), flush=True)
            select.select([descriptor], [], [])
        finally:
            os.close(descriptor)
    predecessor = json.loads((PILOT_QUEUE / 'finished.json').read_text())
    report = json.loads((PILOT_OUT / 'complete.json').read_text())
    assert predecessor['complete'] is True and report['updates'] == 192 and report['full_epochs'] == 3
    assert report['data_scope'] == 'source_label_pilot'
    admission.save('predecessor_complete.json', dict(predecessor=predecessor,
        complete_source_sha256=admission.sha(PILOT_OUT / 'complete.json'),
        raw_AR_quality_accepted=False, reason='Small pilot is not a quality claim; full original labels use a fresh official base, not a false optimizer resume.'))
    staging = json.loads((admission.MODEL / 'local_staging_complete.json').read_text())
    assert staging['complete'] is True and staging['official_content_verified'] is True
    assert staging['revision'] == admission.REV and len(staging['files']) == 25
    for row in staging['files']:
        assert Path(row['name']).name == row['name'] and row['official_content_match'] is True
        path = admission.MODEL / row['name']
        assert path.stat().st_size == row['bytes'] and admission.sha(path) == row['sha256']
    own_cgroup = Path('/proc/self/cgroup').read_text()
    cg = Path('/sys/fs/cgroup/memory')
    headroom = int((cg / 'memory.limit_in_bytes').read_text()) - int((cg / 'memory.usage_in_bytes').read_text())
    mem = {line.split()[0][:-1]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    assert headroom >= 180 * 2**30 and mem['MemAvailable'] >= 180 * 2**20
    gpu_rows = [[value.strip() for value in row.split(',')] for row in subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True).splitlines()]
    assert len(gpu_rows) == 8 and all(int(row[0]) == i and row[1] == admission.UUIDS[i] and int(row[2]) < 2048 and int(row[3]) >= 80000 for i, row in enumerate(gpu_rows))
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            argv = [value.decode() for value in (path / 'cmdline').read_bytes().split(b'\0') if value]
            assert not (str(CORE) in argv and 'train' in argv) and str(ENTRY) not in argv, 'Existing training cannot duplicate'
        except (FileNotFoundError, ProcessLookupError):
            pass
    holders = []
    for row in subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True).splitlines():
        uuid, pid, used = [value.strip() for value in row.split(',')]
        value = admission.identity(int(pid))
        assert value['uids'] == [0] * 4 and value['argv'] == admission.HOLDER and value['cgroup'] == own_cgroup
        assert value['cuda'] in [str(i) for i in range(8)] and uuid == admission.UUIDS[int(value['cuda'])] and int(used) <= 1100
        holders.append(dict(identity=value, uuid=uuid))
    reserves = [Path('/tmp/od_reserve_' + str(i)) for i in range(8)]
    for reserve in reserves:
        assert not reserve.exists() or (reserve.is_file() and not reserve.is_symlink() and reserve.stat().st_uid == os.getuid() and reserve.stat().st_size == 0)
    admission.save('admission.json', dict(gpus=gpu_rows, holders=holders, headroom_bytes=headroom,
        memavailable_bytes=mem['MemAvailable'] * 1024, official_staging_sha256=admission.sha(admission.MODEL / 'local_staging_complete.json')))
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
            assert select.select([fd], [], [], 30)[0], 'Verified holder did not exit; no escalation'
    finally:
        for fd in fds:
            os.close(fd)
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True).strip()
    assert admission.sha(CORE) == CORE_SHA and admission.sha(ENTRY) == args.entry_sha and admission.sha(DATA) == args.metadata_sha
    command = ['env', 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7', 'OMP_NUM_THREADS=2', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'NCCL_IB_DISABLE=1', 'TORCH_NCCL_ASYNC_ERROR_HANDLING=1', '/opt/venv_h3/bin/python', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8', str(ENTRY), 'train', '--metadata', str(DATA), '--model', str(admission.MODEL), '--out', str(OUT)]
    admission.save('command.json', dict(command=command))
    with (QUEUE / 'training.log').open('xb') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        admission.save('live.json', dict(manager_pid=os.getpid(), torchrun_pid=child.pid,
            started_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        print(json.dumps(dict(event='actual_full6000_training_started', torchrun_pid=child.pid, out=str(OUT))), flush=True)
        while True:
            try:
                code = child.wait(timeout=300)
                break
            except subprocess.TimeoutExpired:
                for reserve in reserves:
                    reserve.touch()
    complete = code == 0 and (OUT / 'complete.json').is_file()
    result = dict(exit_code=code, complete=complete, semantic_approval=False)
    if complete:
        result['training_complete'] = json.loads((OUT / 'complete.json').read_text())
    admission.save('finished.json', result)
    print(json.dumps(result), flush=True)
    raise SystemExit(code or (0 if complete else 1))


if __name__ == '__main__':
    main()
