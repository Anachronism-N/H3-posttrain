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
SOURCE = ROOT / 'H3-posttrain/scripts_omnidance/exp/ms/dance_caption_fusion.py'
SOURCE_SHA = 'c74f5e1694f827a180e40e425730e13cdd660775e6756acc7fbbb0c8a29b6f3d'
MANIFEST = MS / 'dance_caption_observed32_30b_manifest_v1.json'
OUT = MS / 'dance_caption_observed32_30b_rewrite_v1'
QUEUE = MS / 'dance_caption_observed32_30b_queue_v1'
GPU = 0
UUID = 'GPU-b3b44a61-0f59-b3e7-268a-d292ff657936'
HOLDER = ['/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/envs/longlive/bin/python', str(ROOT / 'diff/scripts_omnidance/exp/hold_gpu.py')]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(name, value):
    with (QUEUE / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def identity(pid):
    root = Path('/proc') / str(pid)
    status = (root / 'status').read_text()
    uids = [int(v) for line in status.splitlines() if line.startswith('Uid:') for v in line.split()[1:]]
    args = [v.decode() for v in (root / 'cmdline').read_bytes().split(b'\0') if v]
    env = dict(v.split('=', 1) for v in (root / 'environ').read_bytes().decode().split('\0') if '=' in v)
    return dict(pid=pid, uids=uids, argv=args, CUDA_VISIBLE_DEVICES=env.get('CUDA_VISIBLE_DEVICES'),
                start_ticks=(root / 'stat').read_text().rsplit(')', 1)[1].split()[19])


def snapshot():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used', '--format=csv,noheader,nounits'], text=True)
    gpu = next([v.strip() for v in row.split(',')] for row in rows.splitlines() if int(row.split(',')[0]) == GPU)
    assert gpu[1] == UUID and int(gpu[2]) >= 80000
    active = []
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
    for row in processes.splitlines():
        uuid, pid, used = [v.strip() for v in row.split(',')]
        if uuid == UUID:
            active.append(dict(pid=int(pid), used_MiB=int(used), identity=identity(int(pid))))
    return dict(gpu=GPU, uuid=UUID, used_MiB=int(gpu[3]), processes=active)


def main():
    assert os.geteuid() == 0 and MS.is_dir() and MANIFEST.is_file()
    assert sha(SOURCE) == SOURCE_SHA
    assert not OUT.exists() and not QUEUE.exists(), 'Refuse duplicate or overwritten outputs'
    rows = json.loads(MANIFEST.read_text())['records']
    assert len(rows) == 32 and len({r['id'] for r in rows}) == 32
    assert all(r['time_scope']['status'] == 'unknown' for r in rows)
    QUEUE.mkdir(exist_ok=False)
    manifest_sha = sha(MANIFEST)
    mem = {line.split()[0][:-1]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    cgroup = Path('/sys/fs/cgroup/memory')
    headroom = int((cgroup / 'memory.limit_in_bytes').read_text()) - int((cgroup / 'memory.usage_in_bytes').read_text())
    assert mem['MemAvailable'] >= 180 * 2**20 and headroom >= 180 * 2**30
    before = snapshot()
    assert len(before['processes']) <= 1
    reserve = Path('/tmp/od_reserve_0')
    assert reserve.parent.is_dir()
    reserve.touch(exist_ok=True)
    for process in before['processes']:
        value = process['identity']
        descriptor = os.pidfd_open(process['pid'])
        try:
            assert value['uids'] == [0] * 4 and value['argv'] == HOLDER
            assert value['CUDA_VISIBLE_DEVICES'] == str(GPU) and process['used_MiB'] <= 1100
            assert identity(process['pid']) == value and snapshot()['processes'] == [process]
            signal.pidfd_send_signal(descriptor, signal.SIGTERM)
            if not select.select([descriptor], [], [], 10)[0]:
                raise RuntimeError('Exact holder did not terminate; no escalation or unknown-process kill')
        finally:
            os.close(descriptor)
    after = snapshot()
    assert not after['processes'] and after['used_MiB'] < 1024
    command = ['env', 'CUDA_VISIBLE_DEVICES=0', 'OMP_NUM_THREADS=2', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
        '/opt/venv_h3/bin/python', '-u', str(SOURCE), '--rewrite', '--manifest', str(MANIFEST),
        '--model', '/opt/omni30b_dance_official_v1', '--out', str(OUT), '--max-new-tokens', '1536']
    save('plan.json', dict(host='28.216.18.144', manager_pid=os.getpid(), command=command,
        source_sha256=SOURCE_SHA, manifest_sha256=manifest_sha, expected=32,
        reviewer='AI evidence observation; not human', official_revision='26291f793822fb6be9555850f06dfe95f2d7e695',
        frozen_teacher=True, text_only=True, formal_student_training=False,
        observed_media_facts_bound=True, original_rich_MS_source_window_verified=False,
        before_admission=before, after_admission=after, cgroup_headroom_bytes=headroom,
        host_available_bytes=mem['MemAvailable'] * 1024))
    with (QUEUE / 'rewrite.log').open('xb') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        save('live.json', dict(manager_pid=os.getpid(), child_pid=child.pid, started_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        print(json.dumps(dict(event='actual_30b_frozen_rewrite_started', child_pid=child.pid, manager_pid=os.getpid(), out=str(OUT))), flush=True)
        while True:
            try:
                code = child.wait(timeout=300)
                break
            except subprocess.TimeoutExpired:
                reserve.touch(exist_ok=True)
    assert sha(SOURCE) == SOURCE_SHA and sha(MANIFEST) == manifest_sha
    complete = code == 0 and (OUT / 'run_complete.json').is_file()
    result = dict(exit_code=code, complete=complete, expected=32, teacher_only=True, formal_training=False,
                  human_quality_accepted=False, ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if complete:
        report = json.loads((OUT / 'run_complete.json').read_text())
        assert report['complete'] is True and set(report['completed_ids']) == {r['id'] for r in rows}
        result['format_valid'] = sum(json.loads((OUT / (r['id'] + '.json')).read_text())['validation']['format_valid'] for r in rows)
    save('finished.json', result)
    print(json.dumps(result), flush=True)
    if not complete:
        raise SystemExit(code or 1)


if __name__ == '__main__':
    main()
