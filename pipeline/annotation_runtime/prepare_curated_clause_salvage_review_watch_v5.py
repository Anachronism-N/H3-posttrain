"""Read-only fresh N6 GPU3 launch capture; never signal holders or change leases."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import uuid

RT = Path(__file__).resolve().parent
ROOT = RT.parent
spec = importlib.util.spec_from_file_location('prepare_salvage_watch_manager',
    RT / 'curated_clause_salvage_review_watch_resource_manager_v5.py')
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)
base, worker = manager.base, manager.worker
resume = manager.module('salvage_resume_validate_v2', RT / 'curated_clause_salvage_review_resume_validate_v5.py')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--authorize-release', action='store_true')
    p.add_argument('--poll-seconds', type=float, default=20)
    a = p.parse_args()
    if ('28.216.19.225' not in subprocess.check_output(['hostname', '-I'], text=True).split()
            or a.poll_seconds <= 0):
        raise RuntimeError('Read-only capture only on N6 with positive polling')
    config_path = Path(a.out).resolve()
    if config_path.parent != RT or config_path.exists():
        raise RuntimeError('New independent runtime config path required')
    gpu_uuid = base.uuid_for(3)
    running = base.apps()
    processes = [r for r in running if r['uuid'] == gpu_uuid]
    if len(processes) != 1:
        raise RuntimeError('GPU3 must contain exactly one project holder')
    holder = dict(base.identity(processes[0]['pid']), gpu=3, uuid=gpu_uuid)
    base.check_holder(holder, base.identity(holder['pid']), 3, gpu_uuid, running)
    lease = base.lease_snapshot(Path('/tmp/od_reserve_3'))
    suffix = str(time.time_ns()) + '-' + uuid.uuid4().hex[:12]
    capture_path = RT / ('curated_clause_salvage_review_watch_capture_v5-' + suffix + '.json')
    launch_path = RT / ('curated_clause_salvage_review_watch_plan_v5-' + suffix + '.json')
    snapshot = dict(node='N6', gpus=[3], holders=[holder], captured_unix_ns=time.time_ns(),
        signals_sent=False, lease_modified=False,
        lease=None if lease is None else [lease.st_dev, lease.st_ino, lease.st_uid,
            lease.st_mode, lease.st_size, lease.st_mtime_ns], budget=base.headroom())
    worker.write_once(capture_path, snapshot)
    import sys
    sys.path.insert(0, str(RT / 'source_curated_review_v2'))
    import ms_batch_review as review
    resume_path = RT / ('curated_clause_salvage_review_resume_proof_v5-' + suffix + '.json')
    proof = resume.resume_state_proof(worker, review, worker.w.Manifest(worker.pilot.MANIFEST))
    worker.write_once(resume_path, proof)
    launch = dict(schema=worker.PHASE, global_revision=worker.EDITION,
        total_source_count=222939, expected_batch_count=448, input_stream=str(worker.STREAM),
        dynamic_source_watch=True, resume_state_ref=worker.ref(resume_path), imported_previous_summary_ref=worker.ref(worker.PILOT_SUMMARY),
        original_manifest_ref=worker.ref(worker.pilot.MANIFEST), **worker.FLAGS)
    worker.write_once(launch_path, launch)
    stage = worker.checked(dict(path=str(worker.MODEL_REF), sha256=worker.MODEL_REF_SHA))
    modelref = dict(path=str(worker.MODEL_REF), sha256=worker.MODEL_REF_SHA,
        model_identity_sha256=stage['model_identity']['sha256'])
    annotator = ROOT.parent / '32b_replica_stage_receipts/source_reference_v1.json'
    watcher = RT / 'curated_clause_salvage_review_watch_v5.py'
    wrapper = RT / 'curated_clause_salvage_review_watch_resource_manager_v5.py'
    paths = [watcher, wrapper, Path(__file__), RT / 'resource_manager.py',
        RT / 'curated_review_extension_resource_manager_v1.py', worker.PILOT,
        worker.pilot.SUPPORT, worker.RECOVERY, worker.MODEL_REF, worker.pilot.MANIFEST,
        worker.recovery.SELECTED, worker.recovery.REQUESTS, worker.PILOT_SUMMARY,
        launch_path, capture_path, resume_path, annotator, base.HOLDER, worker.BIND, worker.RESUME, Path(resume.__file__)]
    paths += [Path(reference['path']) for reference in resume.references(proof)
        if Path(reference['path']) != worker.OUT / 'state.json']
    paths += [RT / name for name in worker.bind_v2.REPLAY_CODE_SHA]
    paths += sorted((RT / 'source_curated_review_v2').rglob('*.py'))
    for entry, reference in worker.previous_reviews().values():
        paths += [Path(reference['path']), Path(entry['review_receipt']['path']), Path(entry['review_final_ref']['path'])]
    cfg = dict(node='N6', gpus=[3], holders=[holder], release_authorized=a.authorize_release,
        phase=worker.PHASE, queue=str(ROOT / 'queues' / (manager.QUEUE_PREFIX + suffix)),
        out=str(worker.OUT), cwd=str(RT / 'source_curated_review_v2'),
        producer=worker.ref(watcher), plan=worker.ref(launch_path), holder_capture=worker.ref(capture_path),
        input_count=222939, expected_batch_count=448, targetcount=0, dynamic_source_watch=True,
        resume_state_ref=worker.ref(resume_path),
        source_selection=dict(shard=0, nshard=1, limit=0), modelref=modelref,
        source_sha256={str(path): base.sha(path) for path in sorted(set(paths))}, reuse_existing_results=True,
        semantic_approval=False, training_entry_updated=False,
        env=dict(CUDA_VISIBLE_DEVICES='3', OMP_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            MS_VISUAL_CPU_SITEPATH='/opt/venv_eval/lib/python3.11/site-packages'),
        argv=['/opt/venv_vllm/bin/python', '-B', '-u', str(watcher), '--out', str(worker.OUT),
            '--manifest', str(worker.pilot.MANIFEST), '--launch', str(launch_path),
            '--launch-sha', base.sha(launch_path), '--model', worker.MODEL,
            '--model-reference', str(worker.MODEL_REF), '--annotator-model-reference', str(annotator),
            '--ms-source', str(RT / 'source_curated_review_v2'), '--poll-seconds', str(a.poll_seconds), '--watch'])
    if a.authorize_release:
        manager.validate(cfg)
        manager.check_model(cfg)
    worker.write_once(config_path, cfg)
    print(json.dumps(dict(config=str(config_path), config_sha=base.sha(config_path),
        wrapper=str(wrapper), wrapper_sha=base.sha(wrapper), base_manager_sha=base.sha(RT / 'resource_manager.py'),
        watcher_sha=base.sha(watcher), queue=cfg['queue'], launch_ready=a.authorize_release,
        resume_state_ref=worker.ref(resume_path), processed_source_count=proof['processed_source_count'],
        recovered_real_audit_receipt_count=proof['real_audit_receipt_count'],
        required_headroom_bytes=160 * 2**30, signals_sent=False, lease_modified=False)), flush=True)


if __name__ == '__main__':
    main()
