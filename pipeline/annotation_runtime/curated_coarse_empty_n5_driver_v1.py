"""Caller-owned finite runtime authorized by current direct-parent kernel locks."""
import argparse
import fcntl
import os
from pathlib import Path
import stat
import time
import prepare_curated_coarse_empty_n5_v1 as c


def verify_actual_parent(cfg, base, contract, tracker=None, allow_compute=False):
    c.verify_config(cfg, contract)
    resource_ref = c.ref(c.QUEUE / 'admission.json')
    admission = c.checked(resource_ref)
    proof_ref = c.ref(c.QUEUE / 'actual_leases_held.json')
    proof = c.checked(proof_ref)
    launch_ref = c.ref(c.LAUNCH)
    if (proof.get('schema') != 'coarse_empty_n5_actual_leases_held_v1'
            or proof.get('resource_admission_ref') != resource_ref or proof.get('config_ref') != launch_ref
            or proof.get('node') != 'N5' or proof.get('gpus') != c.PAIR):
        raise RuntimeError('actual held-parent proof/config/admission changed')
    expected = {'config_sha256': launch_ref['sha256'], 'manager_sha256': c.BASE_SHA,
                'source_sha256': cfg['source_sha256'], 'producer': cfg['producer'], 'plan': cfg['plan'],
                'model_identity_sha256': cfg['modelref']['model_identity_sha256'], 'node': 'N5',
                'gpus': c.PAIR, 'out': str(c.OUT), 'targetcount': 3}
    if any(admission.get(k) != v for k, v in expected.items()):
        raise RuntimeError('real resource admission config/source SHA mismatch')
    pid = os.getppid()
    parent, own = base.identity(pid), base.identity(os.getpid())
    argv = ['/opt/venv_eval/bin/python', '-B', '-u', str(c.MANAGER), '--config', str(c.LAUNCH),
            '--config-sha', launch_ref['sha256'], '--manager-sha', c.BASE_SHA]
    if (parent['pid'] != admission['manager_identity']['pid'] or parent['uid'] != os.getuid()
            or parent['argv'] != argv or parent['cgroup'] != own['cgroup'] or parent['state'] == 'Z'):
        raise RuntimeError('driver is not direct exact live resource manager child')
    for key in base.IDENTITY_KEYS:
        if parent[key] != admission['manager_identity'][key] or parent[key] != proof['manager_identity'][key]:
            raise RuntimeError('live parent identity changed: ' + key)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1' or os.environ.get('COARSE_EMPTY_N5_CONFIG') != str(c.LAUNCH):
        raise RuntimeError('physical CVD/config env mismatch')
    # observed_unix_ns is historical. Loading may take longer than 120 seconds;
    # fresh lease heartbeats and current FD/kernel FLOCK owners are authoritative.
    if admission['budget']['headroom'] < 160 * 2**30:
        raise RuntimeError('historical admission under 160GiB')
    if len(proof.get('leases', [])) != 2 or [e['gpu'] for e in proof['leases']] != c.PAIR:
        raise RuntimeError('both actual parent leases required')
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
    if len(set(uuids)) != 2:
        raise RuntimeError('duplicate physical GPU UUID')
    running = [app for app in base.apps() if app['uuid'] in set(uuids)]
    if running and not allow_compute:
        raise RuntimeError('unexpected GPU apps before model load; no app signaled')
    if allow_compute:
        if tracker is None:
            raise RuntimeError('model descendant identity tracker required')
        for app in running:
            tracker.verify(app['pid'])
    return {'jobs_manifest_ref': cfg['jobs_manifest_ref'], 'source_job_refs': [h['job_ref'] for h in contract['manifest']['jobs']],
            'out': str(c.OUT), 'global_revision': c.REVISION, 'caller_pid': os.getpid(),
            'tensor_parallel_size': 2, 'gpus': c.PAIR, 'resource_admission_ref': resource_ref,
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
            native_title = value['argv'] in (["VLLM::EngineCore"], ["VLLM::Worker_TP0"], ["VLLM::Worker_TP1"])
            visible_devices_match = value['CVD'] == ['CUDA_VISIBLE_DEVICES=0,1'] or (not value['CVD'] and native_title)
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


class FiniteRuntime:
    def __init__(self, cfg, base, worker, contract):
        self.cfg, self.base, self.worker, self.contract = cfg, base, worker, contract
        self.tracker = ModelDescendants(base)
        self.loaded = None
        self.loader = None
        self.model_started = False
        self.visual_calls = 0
        self.seen_jobs = set()
        self.active_job_ref = None
        self.args = worker.worker_args()

    def verify_current_admission(self, job_refs):
        refs = [h['job_ref'] for h in self.contract['manifest']['jobs']]
        if job_refs != refs or len(refs) != 3:
            raise RuntimeError('current caller admission must bind complete frozen three-job manifest')
        for reference in refs:
            c.checked(reference)
        return verify_actual_parent(self.cfg, self.base, self.contract, self.tracker, self.model_started)

    def __call__(self):
        refs = [h['job_ref'] for h in self.contract['manifest']['jobs']]
        self.verify_current_admission(refs)
        if self.loaded is None:
            if self.model_started:
                raise RuntimeError('shared model load failed; never retry')
            c.require_unreserved(self.contract['jobs'])
            c.verify_local_model(self.cfg, self.worker)
            runtime_root = c.OUT / 'gpu-runtime' / c.REVISION
            runtime_root.mkdir(parents=True, exist_ok=False)
            self.loader = self.worker.bar.runtime_loader(self.args, runtime_root, self.worker.bar.code_identity())
            self.model_started = True
            model, sampling_cls = self.loader()
            self.loaded = (FiniteModel(model, self), sampling_cls)
            self.verify_current_admission(refs)
        return self.loaded


class FiniteModel:
    def __init__(self, model, runtime):
        self.model, self.runtime = model, runtime
        self._ms_bar_model_sha256 = model._ms_bar_model_sha256

    def get_tokenizer(self):
        return self.model.get_tokenizer()

    def chat(self, messages, sampling, use_tqdm=False):
        runtime = self.runtime
        refs = [h['job_ref'] for h in runtime.contract['manifest']['jobs']]
        runtime.verify_current_admission(refs)
        reference = runtime.active_job_ref
        if reference not in refs or len(messages) != 1 or len(sampling) != 1 or runtime.visual_calls >= 3 or reference['path'] in runtime.seen_jobs:
            raise RuntimeError('finite one-visual-per-source/three-total/no-redispatch limit')
        job = c.checked(reference)
        preview = c.load(c.OUT / 'request-previews' / job['source_key'] / job['target_window_id'] / 'request.json')
        if messages != [preview['messages']] or len([v for v in messages[0][1]['content'] if v['type'] == 'image_url']) not in (8, 11):
            raise RuntimeError('only frozen native visual previews, zero text-only calls')
        claim = c.checked(c.ref(c.PRIMARY / 'attempt-budget' / (job['source_key'] + '.json')))
        if claim.get('source_key') != job['source_key'] or claim.get('extra_source_attempt') != 1 or claim.get('maximum_extra_source_attempts') != 1:
            raise RuntimeError('exact canonical source budget claim required')
        runtime.seen_jobs.add(reference['path'])
        runtime.visual_calls += 1  # An ambiguous failed dispatch is spent, not retried.
        return self.model.chat(messages, sampling, use_tqdm=use_tqdm)


def terminal_fallback(worker, hint, status, error):
    path = c.OUT / 'worker-terminals' / (hint['source_key'] + '.json')
    if path.exists():
        return c.ref(path)
    job = c.checked(hint['job_ref'])
    terminal = {'schema': worker.SCHEMA, 'id': job['id'], 'source_key': hint['source_key'],
                'status': status, 'error': str(error), 'global_revision': c.REVISION,
                'original_job_ref': hint['job_ref'], 'joint_complete': False,
                'partial_outputs_preserved': True, 'new_label_review_required': True, **worker.q.FLAGS}
    c.save_once(path, terminal)
    return c.ref(path)


def result_evidence(worker, hint):
    job = c.checked(hint['job_ref'])
    root = c.OUT / 'revisions' / c.REVISION / job['source_key']
    attempt = root / 'windows' / job['target_window_id'] / 'attempt-000001'
    evidence = []
    for name in ('request.json', 'binding.json', 'response.raw.txt', 'generation.json', 'response.json', 'final.json', 'receipt.json'):
        path = attempt / name
        if path.exists():
            evidence.append(c.ref(path))
    for name in ('visual_result.json', 'result.json', 'canonical_budget_claim.json', 'original_job.json'):
        if (root / name).exists():
            evidence.append(c.ref(root / name))
    return evidence


def run(cfg, base, worker, contract):
    # No source reservation or model allocation before real direct-parent validation.
    verify_actual_parent(cfg, base, contract)
    c.require_unreserved(contract['jobs'])
    c.verify_local_model(cfg, worker)
    with (c.OUT / 'independent-worker.lock').open('xb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime = FiniteRuntime(cfg, base, worker, contract)
        admission = runtime.verify_current_admission([h['job_ref'] for h in contract['manifest']['jobs']])
        c.save_once(c.QUEUE / 'actual_worker_admission.json', admission)
        terminals, evidence, stopped = [], [], None
        for hint in contract['manifest']['jobs']:
            if stopped is not None:
                terminals.append(terminal_fallback(worker, hint, 'not_dispatched_after_shared_failure', stopped))
                continue
            runtime.active_job_ref = hint['job_ref']
            try:
                terminal = worker.execute(hint['job_ref'], runtime.args, runtime)
                path = c.OUT / 'worker-terminals' / (hint['source_key'] + '.json')
                if not path.exists():
                    raise RuntimeError('worker did not publish exact current terminal')
                if c.load(path) != terminal:
                    raise RuntimeError('returned actual worker terminal mismatch')
                terminals.append(c.ref(path))
                if terminal.get('joint_result_ref'):
                    worker.q.save_once(c.OUT / 'joint/review-required' / (hint['source_key'] + '.json'),
                        {**terminal, 'result_ref': terminal['joint_result_ref'], 'new_label_review_required': True,
                         'independent_review_status': 'pending_independent_vlm_review'})
                current_evidence = result_evidence(worker, hint)
                evidence.extend(current_evidence)
                if any('dispatch_error.json' == p.name for p in (c.OUT / 'revisions' / c.REVISION / hint['source_key']).rglob('dispatch_error.json')):
                    stopped = 'shared model dispatch failed; subsequent canonical budgets not reserved'
            except Exception as error:
                stopped = repr(error)
                terminals.append(terminal_fallback(worker, hint, 'failed_preserved_no_redispatch', stopped))
                evidence.extend(result_evidence(worker, hint))
        # Record partial/current outputs before any final live-resource check can fail.
        c.save_once(c.QUEUE / 'last_result_proof.json', {'terminal_refs': terminals, 'evidence_refs': evidence,
                    'partial_outputs_preserved': True, 'actual_model_dispatches': runtime.visual_calls,
                    'text_model_calls': 0, 'failure_or_partial_reason': stopped, 'semantic_approval': False})
        # Verify unchanged source inputs and live parent, retaining last results on failure.
        c.frozen(worker, probe=True)
        runtime.verify_current_admission([h['job_ref'] for h in contract['manifest']['jobs']])
        summary = {'schema': 'coarse_empty_n5_finite_stage_summary_v1', 'global_revision': c.REVISION,
                   'jobs_manifest_ref': contract['manifest_ref'], 'source_job_refs': [h['job_ref'] for h in contract['manifest']['jobs']],
                   'source_count': 3, 'terminal_count': len(terminals), 'terminals': terminals,
                   'visual_model_calls': runtime.visual_calls, 'text_model_calls': 0,
                   'evidence_refs': evidence, 'failure_or_partial_reason': stopped,
                   'all_sources_joint_complete_not_required_for_resource_restoration': True,
                   'untargeted_windows_and_style_unchanged': True, 'new_label_review_required': True,
                   'partial_outputs_preserved': True, 'semantic_approval': False, 'training_eligible': False}
        c.save_once(c.OUT / 'finite.stage.summary.json', summary)
        print(c.json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', required=True)
    args = parser.parse_args()
    if Path(args.launch).resolve() != c.LAUNCH:
        raise RuntimeError('only fixed immutable finite launch allowed')
    for key in ('PYTHONPATH', 'PYTHONHOME', 'CONDA_PREFIX'):
        if os.environ.get(key):
            raise RuntimeError('shared Python env injection forbidden for vLLM')
    cfg, worker, base = c.load(c.LAUNCH), c.worker_module(), c.base_module()
    contract = c.frozen(worker, probe=True)
    run(cfg, base, worker, contract)


if __name__ == '__main__':
    main()
