"""Native finite-three caller; current exact parent FD/FLOCK/lease is authoritative."""
import argparse
import os
from pathlib import Path
import stat
import sys
import time
import prepare_coarse3_n6_v1 as c

def verify_actual_parent(cfg, base, contract, tracker=None, allow_compute=False):
    c.verify_config(cfg, contract, verify_history=False)
    resource_ref = c.ref(Path(cfg['queue']) / 'admission.json')
    admission = c.checked(resource_ref)
    proof_ref = c.ref(Path(cfg['queue']) / 'actual_leases_held.json')
    proof = c.checked(proof_ref)
    launch_ref = c.ref(cfg['launch_path'])
    if (proof.get('schema') != 'coarse3_n6_actual_leases_held_v1'
            or proof.get('resource_admission_ref') != resource_ref or proof.get('config_ref') != launch_ref
            or proof.get('node') != 'N6' or proof.get('gpus') != [3]):
        raise RuntimeError('actual held-parent proof/config/admission changed')
    expected = {'config_sha256': launch_ref['sha256'], 'manager_sha256': c.BASE_SHA,
                'source_sha256': cfg['source_sha256'], 'producer': cfg['producer'], 'plan': cfg['plan'],
                'model_identity_sha256': cfg['modelref']['model_identity_sha256'], 'node': 'N6',
                'gpus': [3], 'out': str(c.OUT), 'targetcount': 3}
    if any(admission.get(k) != v for k, v in expected.items()):
        raise RuntimeError('real resource admission config/source SHA mismatch')
    pid = os.getppid()
    parent, own = base.identity(pid), base.identity(os.getpid())
    argv = ['/opt/venv_eval/bin/python', '-B', '-u', str(c.MANAGER), '--config', cfg['launch_path'],
            '--config-sha', launch_ref['sha256'], '--manager-sha', c.BASE_SHA]
    if (parent['pid'] != admission['manager_identity']['pid'] or parent['uid'] != os.getuid()
            or parent['argv'] != argv or parent['cgroup'] != own['cgroup'] or parent['state'] == 'Z'):
        raise RuntimeError('driver is not direct exact live resource manager child')
    for key in base.IDENTITY_KEYS:
        if parent[key] != admission['manager_identity'][key] or parent[key] != proof['manager_identity'][key]:
            raise RuntimeError('live parent identity changed: ' + key)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '3' or os.environ.get('COARSE3_N6_CONFIG') != cfg['launch_path']:
        raise RuntimeError('physical CVD/config env mismatch')
    # observed_unix_ns is historical. Loading may take longer than 120 seconds;
    # fresh lease heartbeats and current FD/kernel FLOCK owners are authoritative.
    if admission['budget']['headroom'] < 160 * 2**30:
        raise RuntimeError('historical admission under 160GiB')
    if len(proof.get('leases', [])) != 1 or len(admission.get('holders', [])) != 1 or [e['gpu'] for e in proof['leases']] != [3]:
        raise RuntimeError('one actual parent lease required')
    lease_refs, uuids = [], []
    for value, holder in zip(proof['leases'], admission['holders']):
        gpu = value['gpu']
        path = Path('/tmp/od_reserve_%d' % gpu)
        if value['path'] != str(path) or value['uuid'] != holder['uuid'] or holder['gpu'] != gpu or base.uuid_for(gpu) != holder['uuid']:
            raise RuntimeError('physical lease path/UUID changed')
        current = path.lstat()
        expected_stat = value['stat']
        c.verify_lease_value(expected_stat)
        if (not stat.S_ISREG(current.st_mode) or current.st_uid != os.getuid() or current.st_size != 0
                or (current.st_dev, current.st_ino) != (expected_stat['st_dev'], expected_stat['st_ino'])
                or not 0 <= time.time_ns() - current.st_mtime_ns <= 60 * 10**9):
            raise RuntimeError('regular empty owned inode/fresh heartbeat changed')
        lock_path = Path('/tmp/ms12k_gpu_%d.lock' % gpu)
        lock = lock_path.lstat()
        if (not stat.S_ISREG(lock.st_mode) or lock.st_uid != os.getuid() or lock.st_size != 0
                or not c.fd_inode_owned(pid, current.st_dev, current.st_ino)
                or not c.fd_inode_owned(pid, lock.st_dev, lock.st_ino)
                or not c.proc_lock_owned(pid, str(lock_path))):
            raise RuntimeError('real parent lease FD and both kernel FLOCK owners required')
        lease_refs.append(c.ref(path))
        uuids.append(holder['uuid'])
    if len(set(uuids)) != 1:
        raise RuntimeError('duplicate physical GPU UUID')
    running = [app for app in base.apps() if app['uuid'] in set(uuids)]
    if running and not allow_compute:
        raise RuntimeError('unexpected GPU apps before model load; no app signaled')
    if allow_compute:
        if tracker is None:
            raise RuntimeError('model descendant identity tracker required')
        for app in running:
            tracker.verify(app['pid'])
    return {'jobs_manifest_ref': cfg['jobs_manifest_ref'], 'source_job_refs': contract['source_job_refs'],
            'out': str(c.OUT), 'global_revision': c.PHASE, 'caller_pid': os.getpid(),
            'tensor_parallel_size': 1, 'gpus': [3], 'resource_admission_ref': resource_ref,
            'actual_held_leases_ref': proof_ref, 'launch_config_ref': launch_ref,
            'current_lease_refs': lease_refs, 'verified_unix_ns': time.time_ns(),
            'semantic_approval': False, 'training_eligible': False}


class ModelDescendants:
    def __init__(self, base):
        self.base = base
        self.own = base.identity(os.getpid())
        self.known = {self.own['pid']: self.own}

    def verify(self, pid):
        chain, seen = [], set()
        current_pid = pid
        while current_pid != self.own['pid']:
            if current_pid in seen or current_pid <= 1:
                raise RuntimeError('unknown GPU app outside tracked driver parent tree; not signaled')
            seen.add(current_pid)
            value = self.base.identity(current_pid)
            native_title = value['argv'] in (["VLLM::EngineCore"], ["VLLM::Worker_TP0"])
            visible_devices_match = value['CVD'] == ['CUDA_VISIBLE_DEVICES=3'] or (not value['CVD'] and native_title)
            if (value['uid'] != self.own['uid'] or value['cgroup'] != self.own['cgroup']
                    or value['exe'] != self.own['exe'] or not visible_devices_match
                    or value['start_ticks'] < self.own['start_ticks'] or value['state'] == 'Z'):
                raise RuntimeError('GPU app UID/start/cgroup/exe/CVD is not own model descendant')
            words = ' '.join(value['argv']).lower()
            if 'vllm' not in words and not ('multiprocessing.spawn' in words and 'spawn_main' in words):
                raise RuntimeError('GPU descendant is not native vLLM/spawn process')
            if current_pid in self.known and any(value[k] != self.known[current_pid][k] for k in self.base.IDENTITY_KEYS):
                raise RuntimeError('tracked model PID identity changed')
            chain.append(value)
            current_pid = value['ppid']
        own = self.base.identity(self.own['pid'])
        if any(own[k] != self.own[k] for k in self.base.IDENTITY_KEYS):
            raise RuntimeError('caller model process identity changed')
        for value in chain:
            self.known[value['pid']] = value



def verify_context(context, plan):
    source = next((s for s in plan['sources'] if s['candidate_ref'] == context.get('candidate_ref')), None)
    c.need(source is not None and context.get('schema') == 'coarse3_filter_recovery_independent_review_v1'
           and context.get('caller_pid') == os.getpid() and context.get('out') == str(c.OUT)
           and context.get('population_ref') == plan['population_ref']
           and context.get('jobs_manifest_ref') == plan['jobs_manifest_ref']
           and context.get('source_job_refs') == plan['source_job_refs']
           and context.get('input_row_sha256') == source['row_sha256']
           and context.get('model_sha256') == c.MODEL_SHA
           and context.get('tensor_parallel_size') == 1 and context.get('maximum_review_calls') == 3,
           'caller context not exact complete source/recovery/label/model binding')
    return source


def run(cfg):
    m, base = c.worker_module(), c.base_module()
    plan = c.source_plan(m)
    c.verify_config(cfg, plan)
    verify_actual_parent(cfg, base, plan)
    c.require_unreviewed(plan)
    c.verify_local_model(cfg)
    lib = m.libraries()
    reviewer = m.make_reviewer(lib, c.MODEL, c.OUT, model_reference=c.MODEL_REF,
                               ffprobe='/opt/ms_tools/ffprobe', ffmpeg='/usr/local/bin/ffmpeg')
    tracker = ModelDescendants(base)

    def admission_callback(context):
        source = verify_context(context, plan)
        proof = verify_actual_parent(cfg, base, plan, tracker, allow_compute=True)
        c.need(getattr(reviewer, '_coarse3_dispatches', 0) <= 3, 'no fourth review allowed')
        for reference in [source['source_job_ref'], source['candidate_ref'], *source['prepared_refs']]:
            c.need(c.sha(reference['path']) == reference['sha256'], 'current job/candidate/native preview changed before dispatch')
        evidence = c.load(source['evidence_ref']['path'])
        root = Path(source['root'])
        for frame in evidence['frames']:
            c.need(c.sha(root / frame['file']) == frame['sha256'], 'current native RGB changed before dispatch')
        for roi in evidence['hand_rois']:
            c.need(c.sha(roi['path']) == roi['sha256'], 'current native ROI changed before dispatch')
        return {**proof, 'admission_verified': True, 'binding': context}

    c.save_once(Path(cfg['queue']) / 'actual_worker_admission.json', verify_actual_parent(cfg, base, plan))
    summary = m.execute(c.OUT, prepare_only=False, reviewer=reviewer, admission_callback=admission_callback,
                        ffprobe='/opt/ms_tools/ffprobe', ffmpeg='/usr/local/bin/ffmpeg')
    rows = [c.json.loads(line) for line in (c.OUT / 'independent.ledger.jsonl').read_bytes().splitlines() if line.strip()]
    c.need(len(rows) == 3 and [r['source_id'] for r in rows] == list(c.IDS), 'exact three independent terminals required')
    report = {'schema': 'coarse3_n6_finite_stage_summary_v1', 'launch_config_ref': c.ref(cfg['launch_path']),
              'population_ref': plan['population_ref'], 'source_count': 3, 'terminal_count': 3,
              'independent_summary_ref': c.ref(c.OUT / 'independent.summary.json'),
              'independent_ledger_ref': c.ref(c.OUT / 'independent.ledger.jsonl'),
              'score_refs': [r['review_receipt'] for r in rows if r['status'] in ('pass', 'confirmed_error', 'uncertain')],
              'review_dispatches': summary['review_dispatches'], 'partial_outputs_preserved': True,
              'review_failed_stays_pending_no_retry': True, 'quality_approval_not_granted': True,
              'semantic_approval': False, 'training_eligible': False}
    c.save_once(c.OUT / 'finite.stage.summary.json', report)
    # Durable partial results survive any subsequent failed live-resource check.
    verify_actual_parent(cfg, base, plan, tracker, allow_compute=True)
    c.verify_stream_prefix(cfg['stream_prefix_proof'])
    print(c.json.dumps(report, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--launch', required=True)
    a = p.parse_args()
    launch = Path(a.launch).resolve()
    c.need(str(launch) == os.environ.get('COARSE3_N6_CONFIG'), 'exact launch environment required')
    c.need(sys.executable == '/opt/venv_vllm/bin/python', 'only original native /opt/venv_vllm interpreter allowed')
    for key in ('PYTHONPATH', 'PYTHONHOME', 'CONDA_PREFIX'):
        c.need(not os.environ.get(key), 'shared Python environment injection forbidden')
    run(c.load(launch))


if __name__ == '__main__':
    main()
