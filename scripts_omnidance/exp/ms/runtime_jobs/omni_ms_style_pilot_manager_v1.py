import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
MS = ROOT / 'diff/logs_omnidance/exp/MS'
SOURCE = ROOT / 'H3-posttrain/scripts_omnidance/exp/ms/omni_dance_sft.py'
DATA = MS / 'omni_ms_style_pilot640_v1/omni.jsonl'
SIDECAR = DATA.parent / 'source_label_sidecar.json'
QUEUE = MS / 'omni_ms_style_pilot_queue_v1'
OUT = MS / 'omni_ms_style_pilot_train_v1'
MODEL = Path('/opt/omni30b_dance_official_v1')
REV = '26291f793822fb6be9555850f06dfe95f2d7e695'
HOLDER = ['/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/envs/longlive/bin/python', str(ROOT / 'diff/scripts_omnidance/exp/hold_gpu.py')]
UUIDS = ['GPU-b3b44a61-0f59-b3e7-268a-d292ff657936', 'GPU-b589db02-1f98-cd20-13ba-4821d10d269c', 'GPU-3e3a11b6-2b3d-fb08-f20d-bf20614cfc61', 'GPU-6162a7de-8566-87d8-4e63-892313f90291', 'GPU-9c565045-f51d-6fdd-326c-fcf754096297', 'GPU-2d0efc0e-c04e-e1e8-53c3-599ba8f51ecb', 'GPU-7d5c8dad-beb0-495a-b3b2-30ac97637289', 'GPU-5b441658-4e57-39d0-4f4d-77bb06d95c74']


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
    status = (root / 'status').read_text()
    env = dict(value.split('=', 1) for value in (root / 'environ').read_bytes().decode().split('\0') if '=' in value)
    return dict(pid=pid, uids=[int(value) for line in status.splitlines() if line.startswith('Uid:') for value in line.split()[1:]],
        argv=[value.decode() for value in (root / 'cmdline').read_bytes().split(b'\0') if value],
        cuda=env.get('CUDA_VISIBLE_DEVICES', env.get('OCCUPY_GPU')),
        cgroup=(root / 'cgroup').read_text(), start_ticks=(root / 'stat').read_text().rsplit(')', 1)[1].split()[19])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--metadata-sha', required=True)
    args = parser.parse_args()
    assert os.geteuid() == 0 and MS.is_dir() and not QUEUE.exists() and not OUT.exists()
    assert sha(SOURCE) == args.source_sha and sha(DATA) == args.metadata_sha
    sidecar = json.loads(SIDECAR.read_text())
    assert sidecar['authorization'] == 'User authorized simple MotionScript + style pilot on 2026-10-23'
    assert sidecar['counts'] == {'train': 512, 'val': 128}
    assert sidecar['semantic_approval'] is False and sidecar['human_review_completed'] is False
    sys.path.insert(0, str(SOURCE.parent))
    from omni_dance_sft import validate_data_boundary
    data_args = argparse.Namespace(metadata=DATA, data_scope='source_label_pilot', smoke_steps=0,
                                   min_epochs=3, max_epochs=3)
    _, training_rows, val_rows, current_data = validate_data_boundary(data_args)
    assert (len(training_rows), len(val_rows)) == (512, 128)
    assert current_data['metadata']['sha256'] == args.metadata_sha
    assert current_data['sidecar']['sha256'] == sha(SIDECAR)
    assert current_data['approval_chain_verified'] is False
    staging = json.loads((MODEL / 'local_staging_complete.json').read_text())
    assert staging['complete'] is True and staging['official_content_verified'] is True
    assert staging['base_model'] == 'Qwen/Qwen3-Omni-30B-A3B-Instruct' and staging['revision'] == REV
    assert staging['destination'] == str(MODEL) and len(staging['files']) == 25
    assert len({row['name'] for row in staging['files']}) == 25
    for row in staging['files']:
        assert Path(row['name']).name == row['name'] and row['official_content_match'] is True
        file = MODEL / row['name']
        assert file.stat().st_size == row['bytes'] and sha(file) == row['sha256'], 'Official staged model changed'
    own_cgroup = Path('/proc/self/cgroup').read_text()
    mem = {row.split()[0][:-1]: int(row.split()[1]) for row in Path('/proc/meminfo').read_text().splitlines()}
    cg = Path('/sys/fs/cgroup/memory')
    headroom = int((cg / 'memory.limit_in_bytes').read_text()) - int((cg / 'memory.usage_in_bytes').read_text())
    assert headroom >= 180 * 2**30 and mem['MemAvailable'] >= 180 * 2**20
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True).splitlines()
    gpus = [[value.strip() for value in row.split(',')] for row in gpu_rows]
    assert len(gpus) == 8 and all(int(row[0]) == i and row[1] == UUIDS[i] and int(row[2]) < 2048 and int(row[3]) >= 80000 for i, row in enumerate(gpus))
    experiments = []
    for root in Path('/proc').iterdir():
        if not root.name.isdigit():
            continue
        try:
            argv = [value.decode() for value in (root / 'cmdline').read_bytes().split(b'\0') if value]
            if any(Path(value).name in ('omni_dance_sft.py', 'omni30b_visual_caption_review.py') for value in argv):
                experiments.append(dict(pid=int(root.name), argv=argv))
        except (FileNotFoundError, ProcessLookupError):
            continue
    assert not experiments, 'Refuse concurrent or duplicate experiment'
    holders = []
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True).splitlines()
    for row in apps:
        uuid, pid, used = [value.strip() for value in row.split(',')]
        value = identity(int(pid))
        assert value['uids'] == [0] * 4 and value['argv'] == HOLDER and value['cgroup'] == own_cgroup
        assert value['cuda'] in [str(i) for i in range(8)] and uuid == UUIDS[int(value['cuda'])] and int(used) <= 1100
        holders.append(dict(uuid=uuid, used_MiB=int(used), identity=value))
    assert len({value['identity']['cuda'] for value in holders}) == len(holders)
    reserves = [Path('/tmp/od_reserve_' + str(i)) for i in range(8)]
    for reserve in reserves:
        assert not reserve.exists() or (reserve.is_file() and not reserve.is_symlink() and reserve.stat().st_uid == os.getuid() and reserve.stat().st_size == 0)
    command = ['env', 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7', 'OMP_NUM_THREADS=2', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'NCCL_IB_DISABLE=1', 'TORCH_NCCL_ASYNC_ERROR_HANDLING=1', '/opt/venv_h3/bin/python', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8', str(SOURCE), 'train', '--metadata', str(DATA), '--model', str(MODEL), '--model-id', 'Qwen/Qwen3-Omni-30B-A3B-Instruct@' + REV, '--out', str(OUT), '--zero-stage1', '--data-scope', 'source_label_pilot', '--smoke-steps', '0', '--min-epochs', '3', '--max-epochs', '3', '--lora-rank', '32', '--lora-alpha', '64', '--lora-lr', '5e-5', '--projector-lr', '1e-5', '--accumulation', '1', '--grad-clip', '1', '--attention', 'sdpa', '--max-length', '8192', '--val-generate', '3', '--max-new-tokens', '512', '--save-every', '100', '--seed', '42']
    QUEUE.mkdir(exist_ok=False)
    save('data_admission.json', dict(complete=True, data_provenance=current_data,
        source_sha256=args.source_sha, counts={'train': 512, 'val': 128}))
    save('plan.json', dict(host='28.216.18.144', command=command, source_sha256=args.source_sha,
        metadata_sha256=args.metadata_sha, sidecar_sha256=sha(SIDECAR), official_revision=REV,
        official_staging_sha256=sha(MODEL / 'local_staging_complete.json'), actual_official_files_verified=25,
        counts={'train': 512, 'val': 128}, epochs=3, expected_optimizer_updates=192,
        label_quality='source_labels_unreviewed', semantic_approval=False, human_review_completed=False,
        optimizer_resume=False, gpus=gpus, holders=holders, headroom_bytes=headroom, memavailable_bytes=mem['MemAvailable'] * 1024))
    for reserve in reserves:
        reserve.touch()
    fds = []
    try:
        for holder in holders:
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
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True).strip()
    assert sha(SOURCE) == args.source_sha and sha(DATA) == args.metadata_sha
    with (QUEUE / 'training.log').open('xb') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        save('live.json', dict(manager_pid=os.getpid(), torchrun_pid=child.pid,
            started_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        print(json.dumps(dict(event='actual_ms_style_30b_pilot_started', manager_pid=os.getpid(), torchrun_pid=child.pid,
            data_scope='source_label_pilot', counts={'train': 512, 'val': 128}, epochs=3, semantic_approval=False)), flush=True)
        while True:
            try:
                code = child.wait(timeout=300)
                break
            except subprocess.TimeoutExpired:
                for reserve in reserves:
                    reserve.touch()
    assert sha(SOURCE) == args.source_sha
    complete = code == 0 and (OUT / 'complete.json').is_file()
    result = dict(exit_code=code, complete=complete, label_quality='source_labels_unreviewed',
        semantic_approval=False, actualhuman=False, formal_reviewed_fusion_training=False)
    if complete:
        report = json.loads((OUT / 'complete.json').read_text())
        assert report['updates'] == 192 and report['full_epochs'] == 3 and report['examples'] == 1536
        result.update(updates=report['updates'], full_epochs=report['full_epochs'], examples=report['examples'])
    save('finished.json', result)
    print(json.dumps(result), flush=True)
    raise SystemExit(code or (0 if complete else 1))


if __name__ == '__main__':
    main()
